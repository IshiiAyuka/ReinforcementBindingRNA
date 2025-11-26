import random
from collections import OrderedDict, defaultdict
import subprocess
import re

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
        indices, prot_list, rna_list = item
        sequences = [["", "", "rna", "prot", rna_list[i], prot_list[i], ""]
                     for i in range(len(prot_list))]

        kwargs = dict(common_kwargs)
        kwargs["gpu_id"] = gpu_id

        results = predict_run(sequences, **kwargs)
        probs = [row[4] for row in results]

        result_q.put((indices, probs))


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

    def score_pairs(self, prot_list, rna_list, device: str = "cpu"):
        N = len(prot_list)
        if N == 0:
            return torch.empty(0, dtype=torch.float32, device=device)

        num_workers = len(self.procs)
        num_jobs = 0
        for w in range(num_workers):
            start = (N * w) // num_workers
            end = (N * (w + 1)) // num_workers
            if start >= end:
                continue
            indices = list(range(start, end))
            p_chunk = prot_list[start:end]
            r_chunk = rna_list[start:end]
            self.task_queues[w].put((indices, p_chunk, r_chunk))
            num_jobs += 1

        out = [None] * N
        for _ in range(num_jobs):
            indices, probs = self.result_q.get()
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
#  RNAfold エネルギー計算
# ===========================
def compute_rnafold_energies(rna_strs, device):
    energies = [0.0] * len(rna_strs)

    non_empty = [(i, s) for i, s in enumerate(rna_strs) if s]
    if not non_empty:
        return torch.tensor(energies, dtype=torch.float32, device=device)

    seqs = [s for _, s in non_empty]
    proc = subprocess.run(
        ["RNAfold", "--noPS"],
        input="\n".join(seqs) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )

    lines = proc.stdout.strip().splitlines()
    for k, (idx, _) in enumerate(non_empty):
        struct_line = lines[2 * k + 1]
        m = re.search(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*\)\s*$", struct_line)
        if m:
            energies[idx] = float(m.group(1))
    return torch.tensor(energies, dtype=torch.float32, device=device)


# ===========================
#  メイン
# ===========================
def main():
    csv_path = "ppi3d.csv"
    weights = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_decoder_AR_1123.pt"
    protein_feat_path = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_3D.pt"
    output_path = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_decoder_AR_reinforce_test_1126_1.pt"

    # --- GPU割り当て ---
    device_ids = [0]
    all_gpus = list(range(torch.cuda.device_count()))
    reward_gpu_ids = [g for g in all_gpus if g not in device_ids]

    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(device_ids[0])

    # --- ハイパーパラメータ ---
    baseline_mean = torch.tensor(0.0, device=device)
    baseline_alpha = 0.5
    max_steps = 10000
    grad_clip_norm = 0.7
    batch_size = 16                    
    entropy_bonus = 0.01
    seed = 42
    OFFTARGET_LAMBDA = 1
    STRUCT_LAMBDA = 1                  # RNAfold 構造スコアの重み
    OFFTARGET_K = 5                     

    # オフターゲット抽出用の乱数（他のrandomと干渉しない）
    rng_off = random.Random(seed + 123)

    # --- データ準備 ---
    df = pd.read_csv(csv_path, low_memory=False)
    df["cluster_id"] = df["s1_binding_site_cluster_data_40_area"].apply(
        lambda x: str(x).split("_")[0]
    )
    cluster_dict = defaultdict(list)
    for _, row in df.iterrows():
        cluster_dict[row["cluster_id"]].append(row["subunit_1"])

    clusters = list(cluster_dict.values())
    random.seed(seed)
    random.shuffle(clusters)
    split_idx = int(0.95 * len(clusters))
    train_ids = {sid for cluster in clusters[:split_idx] for sid in cluster}
    dataset_train = RNADataset_AR(protein_feat_path, csv_path, allowed_ids=train_ids)
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,          
        shuffle=True,
        collate_fn=custom_collate_fn_AR,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )
    torch.manual_seed(seed)

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
        name = k.replace("module.", "")
        new_state_dict[name] = v
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

    # DataLoader から最初の1バッチだけ取得（16個のプール）
    single_batch = next(iter(train_loader))
    protein_feat_all, _, protein_seq_list_all = single_batch

    # ★ オンターゲットを1つ固定（ここでは先頭を固定。必要ならランダム固定にもできる）
    target_idx = 0
    target_prot_seq = [protein_seq_list_all[target_idx]]   # 長さ1のリスト
    off_pool_seqs = [s for i, s in enumerate(protein_seq_list_all) if i != target_idx]

    # 学習に使うのはオンターゲットの特徴量だけ（GPUメモリ節約）
    protein_feat_tgt = protein_feat_all[target_idx:target_idx+1].to(device, non_blocking=True)

    eos_id = config.rna_vocab["<eos>"]
    pad_id = config.rna_vocab["<pad>"]
    sos_id = config.rna_vocab["<sos>"]

    for step in tqdm(range(max_steps), desc="Steps"):
        optimizer.zero_grad(set_to_none=True)

        # ========= サンプリング（オンターゲット1本だけ）=========
        model.eval()
        sampled = sample_decode_multi_AR(
            model,
            protein_feat_tgt,          
            max_len=config.max_len,
            num_samples=1,
            top_k=config.top_k,
            temperature=config.temp,
        )

        B = 1                           
        L = config.max_len
        tokens = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        seq = sampled[0]
        ln = min(len(seq), L)
        if ln > 0:
            tokens[0, :ln] = torch.as_tensor(seq[:ln], dtype=torch.long, device=device)
        if ln < L and ln >= config.min_len:
            tokens[0, ln] = eos_id

        # ========= logits 用 decoder 入力 =========
        sos_col = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        tgt_in = torch.cat([sos_col, tokens[:, :-1]], dim=1)

        # ========= ログ確率用 logits =========
        model.train()
        logits = model(protein_feat_tgt, tgt_in)

        logits = logits.clone()
        logits[..., pad_id] = -1e9
        logits[..., sos_id] = -1e9
        _min_len = min(config.min_len, logits.size(1))
        logits[:, :_min_len, eos_id] = -1e9
        if config.temp != 1.0:
            logits = logits / max(config.temp, 1e-6)

        # ========= 3) LucaOne オンターゲット（1本）=========
        rna_strs = tokens_to_strings(tokens, config.rna_ivocab, eos_id, pad_id, sos_id)
        rna_tgt = rna_strs[0]

        with torch.no_grad():
            R_t = luca_pool.score_pairs(target_prot_seq, [rna_tgt], device=device)  # shape [1]
            R_t = R_t.view(-1)[0]  # scalar

        # ========= 3.1) オフターゲット（残りから5個を毎回ランダム）=========
        k = min(OFFTARGET_K, len(off_pool_seqs))
        off_seqs = rng_off.sample(off_pool_seqs, k=k)
        with torch.no_grad():
            scores_off = luca_pool.score_pairs(off_seqs, [rna_tgt] * k, device=device)  # [k]
            R_off = scores_off.mean() if k > 0 else torch.tensor(0.0, device=device)

        # ========= 3.2) LucaOne 有効報酬 =========
        R_eff = R_t - OFFTARGET_LAMBDA * R_off

        # ========= 3.3) RNAfold 構造スコア（生）=========
        with torch.no_grad():
            E = compute_rnafold_energies([rna_tgt], device=device).view(-1)[0]  # scalar
            length = float(max(len(rna_tgt), 1))
            S_raw = (-E) / length     # scalar（大きいほど安定）

        # 最終報酬（スカラー）
        R_total = R_eff + STRUCT_LAMBDA * S_raw

        # ========= 4) logp（<eos>まで平均）=========
        logp_batch = logprob_from_logits(logits, tokens, pad_id=pad_id, eos_id=eos_id)  # shape [1]
        logp = logp_batch.view(-1)[0]  # scalar

        # ========= 4.1) エントロピー（<eos>まで平均）=========
        logp_all = logits.log_softmax(-1)          # [1, L, V]
        probs = logp_all.exp()
        tok_entropy = -(probs * logp_all).sum(dim=-1)  # [1, L]

        not_pad = (tokens != pad_id)
        eos_mask = (tokens == eos_id)
        csum = eos_mask.int().cumsum(dim=1)
        before_eos = (csum == 0)
        first_eos = eos_mask & (csum == 1)
        include = (before_eos | first_eos) & not_pad

        include_f = include.float()
        den_ent = include_f.sum(dim=-1).clamp_min(1.0)
        entropy = ((tok_entropy * include_f).sum(dim=-1) / den_ent).view(-1)[0]  # scalar

        # ========= 5) REINFORCE損失（オンターゲット1本）=========
        with torch.no_grad():
            baseline_mean = baseline_alpha * baseline_mean + (1 - baseline_alpha) * R_total
        advantage = (R_total - baseline_mean).detach()

        loss = -(advantage * logp) - entropy_bonus * entropy
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        # ====== ログ（オンターゲットのみ）======
        print(
            f"[step] step={step:06d} loss={float(loss.detach().cpu()):.5f} "
            f"R={float(R_t.detach().cpu()):.5f} "
            f"Roff={float(R_off.detach().cpu()):.5f} "
            f"Reff={float(R_eff.detach().cpu()):.5f} "
            f"Sraw={float(S_raw):.5f} "
            f"baseline={float(baseline_mean.detach().cpu()):.5f} "
            f"RNA={rna_tgt}",
            flush=True,
        )

    # 保存
    torch.save(
        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        output_path,
    )
    print(f"[save] Final weights saved to: {output_path}", flush=True)

    luca_pool.close()


if __name__ == "__main__":
    main()
