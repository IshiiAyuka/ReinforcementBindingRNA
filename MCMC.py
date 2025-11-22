import random
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn

from Decoder.model import ProteinToRNA
import Decoder.config as config
from Decoder.dataset import custom_collate_fn, RNADataset_AR
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
#  既存ヘルパ（文字列変換だけ利用）
# ===========================
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
#  MCMC 用: 近傍提案（1 塩基置換）
# ===========================
def propose_tokens_single_mutation(tokens, pad_id, sos_id):
    """
    単純な 1 塩基置換の提案。
    - PAD, <sos> は変更しない
    - 塩基は {A, U, G, C} のみを候補と想定
    """
    tokens_prop = tokens.clone()
    B, L = tokens_prop.shape

    base_ids = [
        config.rna_vocab["A"],
        config.rna_vocab["U"],
        config.rna_vocab["G"],
        config.rna_vocab["C"],
    ]

    for b in range(B):
        # 最大 10 回だけ位置を探す（全部 PAD のような病的ケース対策）
        pos = None
        for _ in range(10):
            cand = random.randint(0, L - 1)
            t = int(tokens_prop[b, cand].item())
            if t != pad_id and t != sos_id:
                pos = cand
                break
        if pos is None:
            continue  # 有効な位置が見つからない場合はスキップ

        cur_tok = int(tokens_prop[b, pos].item())
        # 現在のトークン以外からランダムに選択
        candidates = [t for t in base_ids if t != cur_tok]
        if not candidates:
            continue
        new_tok = random.choice(candidates)
        tokens_prop[b, pos] = new_tok

    return tokens_prop


# ===========================
#  メイン
# ===========================
def main():
    csv_path = "ppi3d.csv"
    weights = "/home/slab/ishiiayuka/M2/Decoder/t30_150M_decoder_AR_100nt_1110.pt"
    protein_feat_path = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M.pt"


    # --- GPU割り当て ---
    device_ids = [0]
    all_gpus = list(range(torch.cuda.device_count()))
    reward_gpu_ids = [g for g in all_gpus if g not in device_ids]
    if len(reward_gpu_ids) == 0:
        reward_gpu_ids = [all_gpus[-1]] if len(all_gpus) > 0 else []

    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(device_ids[0])

    # --- ハイパーパラメータ（MCMC 用） ---
    mcmc_steps     = 10000         # MCMC の反復回数
    OFFTARGET_LAMBDA = 1.0         # R_eff = R - λ * R_off
    beta           = 5.0           # 逆温度（大きいほど高スコアに貪欲）
    seed = 10

    # --- データ準備 ---
    df = pd.read_csv(csv_path, low_memory=False)
    df["cluster_id"] = df["s1_binding_site_cluster_data_40_area"].apply(lambda x: str(x).split("_")[0])
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
        batch_size=8,
        shuffle=True,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    torch.manual_seed(seed)

    # --- モデル定義（初期配列生成にだけ使用） ---
    model = ProteinToRNA(
        input_dim=config.input_dim,
        num_layers=config.num_layers,
        vocab_size=len(config.rna_vocab),
        max_len=config.max_len
    )
    state_dict = torch.load(weights, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

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

    # --- 報酬プール初期化 ---
    luca_pool = LucaPPIRewardPool(gpu_ids=reward_gpu_ids, common_kwargs=common_kwargs)

    # DataLoader から最初の1バッチだけ取得し、それを MCMC で探索
    single_batch = next(iter(train_loader))
    protein_feat, tgt_seqs, protein_seq_list = single_batch
    protein_feat = protein_feat.to(device, non_blocking=True)

    eos_id  = config.rna_vocab["<eos>"]
    pad_id  = config.rna_vocab["<pad>"]
    sos_id  = config.rna_vocab["<sos>"]

    # ========= 初期配列（モデル＋サンプリング）=========
    was_training = model.training
    model.eval()
    sampled = sample_decode_multi_AR(
        model,
        protein_feat,
        max_len=config.max_len,
        num_samples=1,
        top_k=config.top_k,
        temperature=config.temp,
    )  # List[List[int]]（<eos>以降なし）
    if was_training:
        model.train()

    # PAD 埋めで [B, L] の tokens を作成
    B = protein_feat.size(0)
    L = config.max_len
    tokens = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(sampled):
        ln = min(len(seq), L)
        if ln > 0:
            tokens[i, :ln] = torch.as_tensor(seq[:ln], dtype=torch.long, device=device)
        if ln < L and ln >= config.min_len:
            tokens[i, ln] = eos_id

    # ========= 初期状態の Reff を計算 =========
    rna_strs = tokens_to_strings(tokens, config.rna_ivocab, eos_id, pad_id, sos_id)
    with torch.no_grad():
        # オンターゲット
        R = luca_pool.score_pairs(protein_seq_list, rna_strs, device=device)  # [B]
        if R.dim() == 0:
            R = R.expand(tokens.size(0))

        # オフターゲット（クロスバッチ）
        B_batch = len(protein_seq_list)
        if B_batch > 1:
            prot_off, rna_off = [], []
            for i in range(B_batch):
                for j in range(B_batch):
                    if j == i:
                        continue
                    prot_off.append(protein_seq_list[j])
                    rna_off.append(rna_strs[i])
            scores_off = luca_pool.score_pairs(prot_off, rna_off, device=device)
            R_off = scores_off.reshape(B_batch, B_batch - 1).mean(dim=1)  # [B]
        else:
            R_off = torch.zeros_like(R)

    R_eff = R - OFFTARGET_LAMBDA * R_off  # [B]

    # 各タンパク質ごとのベストを記録
    best_tokens = tokens.clone()
    best_Reff   = R_eff.clone()

    print("[init] R_mean={:.5f} Roff_mean={:.5f} Reff_mean={:.5f}".format(
        float(R.mean().detach().cpu()),
        float(R_off.mean().detach().cpu()),
        float(R_eff.mean().detach().cpu()),
    ), flush=True)

    # ========= MCMC ループ =========
    for step in tqdm(range(mcmc_steps), desc="MCMC"):
        # 提案
        tokens_prop = propose_tokens_single_mutation(tokens, pad_id=pad_id, sos_id=sos_id)

        # 提案配列のスコア計算
        rna_strs_prop = tokens_to_strings(tokens_prop, config.rna_ivocab, eos_id, pad_id, sos_id)
        with torch.no_grad():
            # オンターゲット
            R_prop = luca_pool.score_pairs(protein_seq_list, rna_strs_prop, device=device)
            if R_prop.dim() == 0:
                R_prop = R_prop.expand(tokens_prop.size(0))

            # オフターゲット
            if B_batch > 1:
                prot_off_p, rna_off_p = [], []
                for i in range(B_batch):
                    for j in range(B_batch):
                        if j == i:
                            continue
                        prot_off_p.append(protein_seq_list[j])
                        rna_off_p.append(rna_strs_prop[i])
                scores_off_p = luca_pool.score_pairs(prot_off_p, rna_off_p, device=device)
                R_off_prop = scores_off_p.reshape(B_batch, B_batch - 1).mean(dim=1)
            else:
                R_off_prop = torch.zeros_like(R_prop)

        R_eff_prop = R_prop - OFFTARGET_LAMBDA * R_off_prop  # [B]

        # 受理確率（Metropolis-Hastings, 対称提案を仮定）
        delta = R_eff_prop - R_eff  # [B]
        # β をかけた差の指数 → クリップして数値安定化
        with torch.no_grad():
            accept_prob = torch.exp(beta * delta).clamp(max=1.0)  # [B]
            u = torch.rand_like(accept_prob)
            accept_mask = (u < accept_prob)

        # 受理されたものだけ更新
        if accept_mask.any():
            idx = accept_mask.nonzero(as_tuple=True)[0]
            tokens[idx]   = tokens_prop[idx]
            R[idx]        = R_prop[idx]
            R_off[idx]    = R_off_prop[idx]
            R_eff[idx]    = R_eff_prop[idx]

            # ベスト更新（各サンプル毎）
            better_mask = R_eff > best_Reff
            if better_mask.any():
                bidx = better_mask.nonzero(as_tuple=True)[0]
                best_Reff[bidx]   = R_eff[bidx]
                best_tokens[bidx] = tokens[bidx]

        # ログ用のスカラー
        R_mean_val     = float(R.mean().detach().cpu())
        Roff_mean_val  = float(R_off.mean().detach().cpu())
        Reff_mean_val  = float(R_eff.mean().detach().cpu())
        acc_rate       = float(accept_mask.float().mean().detach().cpu())

        # 現在の tokens から RNA 配列を復元して | で連結
        rna_strs_step = tokens_to_strings(tokens, config.rna_ivocab, eos_id, pad_id, sos_id)
        rna_line = "|".join(rna_strs_step)

        print(
            f"[step] step={step:06d} "
            f"R={R_mean_val:.5f} Roff={Roff_mean_val:.5f} Reff={Reff_mean_val:.5f} "
            f"acc={acc_rate:.3f} RNAs={rna_line}",
            flush=True,
        )


    # プール終了
    luca_pool.close()


if __name__ == "__main__":
    main()
