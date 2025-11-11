import random
import random as pyrand
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from Decoder.model import ProteinToRNA_NAR
import Decoder.config as config
from Decoder.dataset import custom_collate_fn, RNADataset_NAR
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from Decoder.decode import sample_decode_multi

from LucaOneTasks.src.predict_v1 import run as predict_run
import multiprocessing as mp


# ===========================
#  報酬: マルチGPU並列プール
# ===========================
def _reward_worker(task_q, result_q, gpu_id: int, common_kwargs: dict):

    try:
        torch.cuda.set_device(gpu_id)
    except Exception:
        pass  # CPUフォールバック許容

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
    """
    マルチGPUで predict_run を並列化する常駐プール。
    """
    def __init__(self, gpu_ids, common_kwargs, max_workers=None):
        # CUDAと相性の良い 'spawn'
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
        """
        prot_list, rna_list: 同長のリスト（任意長）
        → 並列に分割してスコアリングし、元の順序で torch.Tensor 返却
        """
        N = len(prot_list)
        if N == 0:
            return torch.empty(0, dtype=torch.float32, device=device)

        num_workers = len(self.procs)
        chunks = []
        for w in range(num_workers):
            start = (N * w) // num_workers
            end   = (N * (w + 1)) // num_workers
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
    not_pad  = (tokens != pad_id)
    eos_mask = (tokens == eos_id)
    csum     = eos_mask.int().cumsum(dim=1)
    before_eos = (csum == 0)
    first_eos  = eos_mask & (csum == 1)
    include = (before_eos | first_eos) & not_pad
    num = (logp_tok * include.float()).sum(dim=-1)
    den = include.float().sum(dim=-1).clamp_min(1.0)
    return num / den  # 長さ正規化（平均）

def tokens_to_strings(tokens, ivocab, eos_id, pad_id, sos_id):
    seqs = []
    for row in tokens.tolist():
        s = []
        for t in row:
            if t == eos_id: break
            if t == pad_id or t == sos_id: continue
            s.append(ivocab[int(t)])
        seqs.append("".join(s))
    return seqs

# ===========================
#  メイン
# ===========================
def main():
    csv_path = "ppi3d.csv"
    weights = "/home/slab/ishiiayuka/M2/Decoder/t30_150M_decoder_NAR_100nt_1106.pt"
    protein_feat_path = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M.pt"
    output_path = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_decoder_NAR_after_reinforce_ppi3d_1111_test.pt"

    # --- GPU割り当て ---
    device_ids = [0]
    all_gpus = list(range(torch.cuda.device_count()))
    reward_gpu_ids = [g for g in all_gpus if g not in device_ids]
    if len(reward_gpu_ids) == 0:
        reward_gpu_ids = [all_gpus[-1]] if len(all_gpus) > 0 else []

    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(device_ids[0])

    # --- ハイパーパラメータ ---
    baseline_mean   = torch.tensor(0.0, device=device)  # 報酬の移動平均(=ベースライン)
    baseline_alpha  = 0.7
    max_steps       = 10000
    grad_clip_norm  = 0.7
    batch_size      = 8
    entropy_bonus = 0.01

    # --- オフターゲット抑制（シンプル） ---
    OFFTARGET_LAMBDA = 1   # R_eff = R - λ * R_off

    # --- データ準備 ---
    df = pd.read_csv(csv_path, low_memory=False)
    df["cluster_id"] = df["s1_binding_site_cluster_data_40_area"].apply(lambda x: str(x).split("_")[0])
    cluster_dict = defaultdict(list)
    for _, row in df.iterrows():
        cluster_dict[row["cluster_id"]].append(row["subunit_1"])

    clusters = list(cluster_dict.values())
    pyrand.seed(42)
    pyrand.shuffle(clusters)
    split_idx = int(0.95 * len(clusters))
    train_ids = {sid for cluster in clusters[:split_idx] for sid in cluster}
    dataset_train = RNADataset_NAR(protein_feat_path, csv_path, allowed_ids=train_ids)
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    torch.manual_seed(42)

    # --- モデル定義 ---
    model = ProteinToRNA_NAR(input_dim=config.input_dim, num_layers=config.num_layers)
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
        model_path = "LucaOneTasks/",
        llm_truncation_seq_length = 100,
        dataset_name = "ncRPI",
        dataset_type = "gene_protein",
        task_type = "binary_class",
        task_level_type = "seq_level",
        model_type = "lucappi2",
        input_type = "matrix",
        input_mode = "pair",
        threshold = 0.5,
        step = "716380",
        time_str = "20240404105148",
        topk = None,
        gpu_id = reward_gpu_ids[0] if len(reward_gpu_ids) > 0 else 0,  # 初期値（各ワーカーで上書き）
        emb_dir = None,
        matrix_embedding_exists = False
    )

    # --- 報酬プール初期化（GPU[2,3]） ---
    luca_pool = LucaPPIRewardPool(gpu_ids=reward_gpu_ids, common_kwargs=common_kwargs)

    # DataLoader から最初の1バッチだけ取得し、それを max_steps 回使い回す
    single_batch = next(iter(train_loader))
    protein_feat, tgt_seqs, protein_seq_list = single_batch
    protein_feat = protein_feat.to(device, non_blocking=True)

    for step in tqdm(range(max_steps), desc="Steps"):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        eos_id  = config.rna_vocab_NAR["<eos>"]
        pad_id  = config.rna_vocab_NAR["<pad>"]
        sos_id  = config.rna_vocab_NAR["<sos>"]
        mask_id = config.rna_vocab_NAR["<MASK>"]

        # ========= サンプリング（sample_decode_multiで生成）=========
        with torch.no_grad():
            was_training = model.training
            model.eval()
            sampled = sample_decode_multi(
                model,
                protein_feat,
                max_len=config.max_len,
                num_samples=1,
                top_k=config.top_k,
                temperature=config.temp,
            )  # List[List[int]]（<eos>以降なし）
            if was_training:
                model.train()

        # PAD埋めで [B, L] の tokens を作成
        B = protein_feat.size(0)
        L = config.max_len
        tokens = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(sampled):
            ln = min(len(seq), L)
            if ln > 0:
                tokens[i, :ln] = torch.as_tensor(seq[:ln], dtype=torch.long, device=device)
            if ln < L and ln >= config.min_len:
                tokens[i, ln] = eos_id

        # ========= 並列forward（損失用；サンプルと同条件）=========
        loss_was_training = model.training
        model.eval()
        if isinstance(model, nn.DataParallel):
            logits = model.module.forward_parallel(protein_feat, out_len=L)  # [B, L, V]
        else:
            logits = model.forward_parallel(protein_feat, out_len=L)
        if loss_was_training:
            model.train()

        logits = logits.clone()
        logits[..., pad_id]  = -1e9
        logits[..., sos_id]  = -1e9
        logits[..., mask_id] = -1e9
        _min_len = min(config.min_len, logits.size(1))
        logits[:, :_min_len, eos_id] = -1e9

        if config.temp != 1.0:
            logits = logits / max(config.temp, 1e-6)

        N, L_, V = logits.shape
        flat = logits.view(N * L_, V)
        if 0 < config.top_k < V:
            topv, topi = torch.topk(flat, k=config.top_k, dim=-1)
            masked = torch.full_like(flat, -1e9)
            masked.scatter_(1, topi, topv)
            flat = masked
        logits = flat.view(N, L_, V)

        # ========= 3) オンターゲット（並列スコア）=========
        rna_strs = tokens_to_strings(tokens, config.rna_ivocab_NAR, eos_id, pad_id, sos_id)
        with torch.no_grad():
            R = luca_pool.score_pairs(protein_seq_list, rna_strs, device=device)  # [B]
            if R.dim() == 0:
                R = R.expand(tokens.size(0))

        # ========= 3.5) オフターゲット（クロスバッチ平均）=========
        B_batch = len(protein_seq_list)
        if B_batch > 1:
            prot_off, rna_off = [], []
            for i in range(B_batch):
                for j in range(B_batch):
                    if j == i:
                        continue
                    prot_off.append(protein_seq_list[j])
                    rna_off.append(rna_strs[i])
            with torch.no_grad():
                scores_off = luca_pool.score_pairs(prot_off, rna_off, device=device)
            R_off = scores_off.reshape(B_batch, B_batch - 1).mean(dim=1)  # [B]
        else:
            R_off = torch.zeros_like(R)

        # ========= 3.6) 有効報酬 =========
        R_eff = R - OFFTARGET_LAMBDA * R_off

        # ========= 4) logp（<eos>まで平均）=========
        logp_batch = logprob_from_logits(logits, tokens, pad_id=pad_id, eos_id=eos_id)

        # ========= 5) REINFORCE損失 =========
        with torch.no_grad():
            R_mean = R_eff.mean()
            baseline_mean = baseline_alpha * baseline_mean + (1 - baseline_alpha) * R_mean
        advantage = (R_eff - baseline_mean).detach()
        loss = -(advantage * logp_batch).mean()

        # 逆伝播＆更新（← 1ステップにつき1回）
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        # ====== ログ（エントロピー関連は出力しない） ======
        loss_val = float(loss.detach().cpu())
        R_mean_val = float(R.mean().detach().cpu())
        Roff_mean_val = float(R_off.mean().detach().cpu())
        Reff_mean_val = float(R_eff.mean().detach().cpu())
        base_val = float(baseline_mean.detach().cpu())
        print(
            f"[step] step={step:06d} loss={loss_val:.5f} R={R_mean_val:.5f} "
            f"Roff={Roff_mean_val:.5f} Reff={Reff_mean_val:.5f} baseline={base_val:.5f}",
            flush=True,
        )

     # 保存
    final_path = f"{output_path}"
    torch.save(
        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        final_path
    )
    print(f"[save] Final weights saved to: {final_path}", flush=True)

    # プール終了
    luca_pool.close()


if __name__ == "__main__":
    main()
