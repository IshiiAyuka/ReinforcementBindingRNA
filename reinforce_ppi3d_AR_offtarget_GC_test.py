import random
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from Decoder.model import ProteinToRNA
import Decoder.config as config
from Decoder.dataset import custom_collate_fn_AR, RNADataset_AR
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from Decoder.decode import sample_decode_multi_AR

from LucaOneTasks.src.predict_v1 import run as predict_run
import multiprocessing as mp


# ===========================
#  報酬: マルチGPU並列プール
# ===========================
def _reward_worker(task_q, result_q, gpu_id: int, common_kwargs: dict):
    while True:
        item = task_q.get()
        if item is None:
            break
        job_id, indices, prot_list, rna_list = item
        sequences = [["", "", "rna", "prot", rna_list[i], prot_list[i], ""] for i in range(len(prot_list))]

        kwargs = dict(common_kwargs)
        kwargs["gpu_id"] = gpu_id  # 強制上書き

        results = predict_run(sequences, **kwargs)
        probs = [row[4] for row in results]  # luca の5列目を確率と解釈

        result_q.put((job_id, indices, probs))


class LucaPPIRewardPool:
    def __init__(self, gpu_ids, common_kwargs, max_workers=None):
        self.ctx = mp.get_context("spawn")
        self.gpu_ids = list(gpu_ids) if gpu_ids else [0]
        if max_workers is None:
            max_workers = len(self.gpu_ids)

        self.task_queues = []
        self.procs = []
        self.result_q = self.ctx.Queue()

        for i in range(max_workers):
            tq = self.ctx.Queue(maxsize=2)
            self.task_queues.append(tq)
            p = self.ctx.Process(
                target=_reward_worker,
                args=(tq, self.result_q, self.gpu_ids[i % len(self.gpu_ids)], common_kwargs),
                daemon=True
            )
            p.start()
            self.procs.append(p)

        self._next_job_id = 0

    def score_pairs(self, prot_list, rna_list, device: str = "cpu"):
        N = len(prot_list)
        if N == 0:
            return torch.empty(0, dtype=torch.float32, device=device)

        num_workers = len(self.procs)
        chunks = []
        for w in range(num_workers):
            start = (N * w) // num_workers
            end = (N * (w + 1)) // num_workers
            if start >= end:
                continue
            indices = list(range(start, end))
            p_chunk = prot_list[start:end]
            r_chunk = rna_list[start:end]
            job_id = self._next_job_id
            self._next_job_id += 1
            self.task_queues[w].put((job_id, indices, p_chunk, r_chunk))
            chunks.append(job_id)

        out = [None] * N
        for _ in range(len(chunks)):
            job_id, indices, probs = self.result_q.get()
            for idx, val in zip(indices, probs):
                out[idx] = val

        return torch.tensor(out, dtype=torch.float32, device=device)

    def close(self):
        for q in self.task_queues:
            q.put(None)
        for p in self.procs:
            p.join(timeout=5)


# ===========================
#  既存ヘルパ
# ===========================
def logprob_from_logits(logits, tokens, pad_id, eos_id):
    logp_tok = logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # [B,L]
    not_pad = (tokens != pad_id)
    eos_mask = (tokens == eos_id)
    csum = eos_mask.int().cumsum(dim=1)
    before_eos = (csum == 0)
    first_eos = eos_mask & (csum == 1)
    include = (before_eos | first_eos) & not_pad
    num = (logp_tok * include.float()).sum(dim=-1)
    den = include.float().sum(dim=-1).clamp_min(1.0)
    return num / den  # 長さ正規化（平均）


def tokens_to_strings(tokens, ivocab, eos_id, pad_id, sos_id):
    seqs = []
    for row in tokens.tolist():
        s = []
        for t in row:
            if t == eos_id:
                break
            if t == pad_id or t == sos_id:
                continue
            s.append(ivocab[int(t)])
        seqs.append("".join(s))
    return seqs


# ===========================
#  GC含量ヘルパ
# ===========================
def compute_gc_fraction(seq: str) -> float:
    """A/U/G/C のうち G,C の比率を返す。対象塩基が無ければ 0."""
    if not seq:
        return 0.0
    bases = [c for c in seq if c in ("A", "U", "G", "C")]
    if not bases:
        return 0.0
    gc_count = sum(1 for c in bases if c in ("G", "C"))
    return gc_count / len(bases)


# ===========================
#  メイン
# ===========================
def main():
    csv_path = "ppi3d.csv"
    weights = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_decoder_AR_1123.pt"
    protein_feat_path = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_3D.pt"
    output_path = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_decoder_AR_reinforce_test_1126_2.pt"

    # --- GPU割り当て ---
    device_ids = [3]
    all_gpus = list(range(torch.cuda.device_count()))
    reward_gpu_ids = [g for g in all_gpus if g not in device_ids]

    device = f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device_ids[0])

    # --- ハイパーパラメータ ---
    baseline_mean = torch.tensor(0.0, device=device)
    baseline_alpha = 0.5
    max_steps = 10000
    grad_clip_norm = 0.7
    pool_batch_size = 16          # 固定プール（最初の1バッチ）
    gen_batch_size = 6            # ★ 毎stepの生成は 1(on) + 5(off) = 6
    entropy_bonus = 0.01
    seed = 42

    OFFTARGET_LAMBDA = 1.0
    OFFTARGET_K = 5              # ★ 毎stepランダムに5本

    GC_LOW = 0.40
    GC_HIGH = 0.80
    GC_LAMBDA = 1.0

    random.seed(seed)
    torch.manual_seed(seed)
    rng_off = random.Random(seed + 123)  # オフターゲット選択専用（他と干渉させない）

    # --- データ準備 ---
    df = pd.read_csv(csv_path, low_memory=False)
    df["cluster_id"] = df["s1_binding_site_cluster_data_40_area"].apply(lambda x: str(x).split("_")[0])
    cluster_dict = defaultdict(list)
    for _, row in df.iterrows():
        cluster_dict[row["cluster_id"]].append(row["subunit_1"])

    clusters = list(cluster_dict.values())
    random.shuffle(clusters)
    split_idx = int(0.95 * len(clusters))
    train_ids = {sid for cluster in clusters[:split_idx] for sid in cluster}

    dataset_train = RNADataset_AR(protein_feat_path, csv_path, allowed_ids=train_ids)
    train_loader = DataLoader(
        dataset_train,
        batch_size=pool_batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn_AR,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,  # 16本を確実に確保
    )

    # --- モデル定義 ---
    model = ProteinToRNA(
        input_dim=config.input_dim,
        num_layers=config.num_layers,
        vocab_size=len(config.rna_vocab),
        max_len=config.max_len,
    )
    state_dict = torch.load(weights, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # --- LucaOne 共通引数（プールへ渡す） ---
    common_kwargs = dict(
        model_path="LucaOneTasks/",
        llm_truncation_seq_length=100,
        dataset_name="ncRPI",
        dataset_type="gene_protein",
        task_type="binary_class",
        task_level_type="seq_level",
        model_type="lucappi2",
        input_type="matrix",
        input_mode="pair",
        threshold=0.5,
        step="716380",
        time_str="20240404105148",
        topk=None,
        gpu_id=reward_gpu_ids[0] if len(reward_gpu_ids) > 0 else 0,
        emb_dir=None,
        matrix_embedding_exists=False,
    )

    # --- 報酬プール初期化 ---
    luca_pool = LucaPPIRewardPool(gpu_ids=reward_gpu_ids, common_kwargs=common_kwargs)

    # ===== 固定プールを1回だけ作る（16本）=====
    single_batch = next(iter(train_loader))
    protein_feat_all, _, protein_seq_list_all = single_batch  # [16, ...], len=16

    eos_id = config.rna_vocab["<eos>"]
    pad_id = config.rna_vocab["<pad>"]
    sos_id = config.rna_vocab["<sos>"]

    target_idx = 0
    remain = [i for i in range(len(protein_seq_list_all)) if i != target_idx]


    for step in tqdm(range(max_steps), desc="Steps"):
        optimizer.zero_grad(set_to_none=True)

        # ============================================================
        # ★ 毎step: (on=1) + (off=5をランダム) の計6本だけで生成する
        # ============================================================
        k = min(OFFTARGET_K, len(remain))
        off_indices = rng_off.sample(remain, k=k) if k > 0 else []
        sel_indices = [target_idx] + off_indices  # 先頭がオンターゲット
        # Bは原則6（remain>=5なら）
        B = len(sel_indices)

        protein_feat_step = protein_feat_all[sel_indices].to(device, non_blocking=True)  # [B, ...]
        prot_step_list = [protein_seq_list_all[i] for i in sel_indices]
        target_prot_step = prot_step_list[0]
        off_prots_step = prot_step_list[1:]  # len=k

        # ===== サンプリング（B本生成）=====
        model.eval()
        sampled = sample_decode_multi_AR(
            model,
            protein_feat_step,
            max_len=config.max_len,
            num_samples=1,
            top_k=config.top_k,
            temperature=config.temp,
        )
        # sampled: list length B

        # ===== tokens 化 =====
        L = config.max_len
        tokens = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(sampled):
            ln = min(len(seq), L)
            if ln > 0:
                tokens[i, :ln] = torch.as_tensor(seq[:ln], dtype=torch.long, device=device)
            if ln < L and ln >= config.min_len:
                tokens[i, ln] = eos_id

        # ===== logits 用 decoder 入力 =====
        sos_col = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        tgt_in = torch.cat([sos_col, tokens[:, :-1]], dim=1)

        # ===== ログ確率用 logits（gradあり）=====
        logits = model(protein_feat_step, tgt_in)  # model.eval() でも grad は流れる（dropoutだけOFF）
        logits = logits.clone()
        logits[..., pad_id] = -1e9
        logits[..., sos_id] = -1e9
        _min_len = min(config.min_len, logits.size(1))
        logits[:, :_min_len, eos_id] = -1e9
        if config.temp != 1.0:
            logits = logits / max(config.temp, 1e-6)

        # ===== 文字列化（B本）=====
        rna_strs = tokens_to_strings(tokens, config.rna_ivocab, eos_id, pad_id, sos_id)

        # ============================================================
        # ★ 報酬は「オンターゲット生成（0番RNA）」だけでスカラーにする
        # ============================================================
        rna_on = rna_strs[0]

        with torch.no_grad():
            Ron = luca_pool.score_pairs([target_prot_step], [rna_on], device=device).view(-1)[0]  # scalar
            if len(off_prots_step) > 0:
                Roff = luca_pool.score_pairs(off_prots_step, [rna_on] * len(off_prots_step), device=device).mean()
            else:
                Roff = torch.tensor(0.0, device=device)

        Reff = Ron - OFFTARGET_LAMBDA * Roff

        # ===== GC（罰則付き、オンターゲットRNAのみ）=====
        gc_on = torch.tensor(compute_gc_fraction(rna_on), dtype=torch.float32, device=device)  # scalar
        ok = (gc_on >= GC_LOW) & (gc_on <= GC_HIGH)
        Rgc = torch.where(ok, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))  # scalar

        R_total = Reff + GC_LAMBDA * Rgc  # scalar

        # ===== logp（オンターゲットの1本だけ）=====
        logp_batch = logprob_from_logits(logits, tokens, pad_id=pad_id, eos_id=eos_id)  # [B]
        logp_on = logp_batch[0]  # scalar

        # ===== エントロピー（オンターゲット1本だけ）=====
        logp_all = logits.log_softmax(-1)              # [B, L, V]
        probs = logp_all.exp()
        tok_entropy = -(probs * logp_all).sum(dim=-1)  # [B, L]

        not_pad = (tokens != pad_id)
        eos_mask = (tokens == eos_id)
        csum = eos_mask.int().cumsum(dim=1)
        before_eos = (csum == 0)
        first_eos = eos_mask & (csum == 1)
        include = (before_eos | first_eos) & not_pad

        include_f = include.float()
        den_ent = include_f.sum(dim=-1).clamp_min(1.0)                  # [B]
        entropy_batch = (tok_entropy * include_f).sum(dim=-1) / den_ent # [B]
        entropy_on = entropy_batch[0]                                   # scalar

        # ===== REINFORCE（スカラー）=====
        with torch.no_grad():
            baseline_mean = baseline_alpha * baseline_mean + (1 - baseline_alpha) * R_total
        advantage = (R_total - baseline_mean).detach()

        loss = -(advantage * logp_on) - entropy_bonus * entropy_on
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        # ===== ログ（オンターゲットだけ、1行）=====
        print(
            f"[step] step={step:06d} "
            f"loss={float(loss.detach().cpu()):.5f} "
            f"Ron={float(Ron.detach().cpu()):.5f} "
            f"Roff={float(Roff.detach().cpu()):.5f} "
            f"Reff={float(Reff.detach().cpu()):.5f} "
            f"Rgc={float(Rgc.detach().cpu()):+.1f} "
            f"GC={float(gc_on.detach().cpu()):.3f} "
            f"baseline={float(baseline_mean.detach().cpu()):.5f} "
            f"RNA={rna_on}",
            flush=True,
        )

    # 保存
    torch.save(
        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        output_path
    )
    print(f"[save] Final weights saved to: {output_path}", flush=True)

    luca_pool.close()


if __name__ == "__main__":
    main()
