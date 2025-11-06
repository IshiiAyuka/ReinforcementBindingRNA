import torch
import os
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from Decoder.decode import sample_decode_reinforce
from Decoder.model import ProteinToRNA_reinforce
import Decoder.config as config
import pandas as pd
from Decoder.dataset import RNADataset, custom_collate_fn, DeepCLIPProteinDataset
import random
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Optional
from collections import OrderedDict
from LucaOneTasks.src.predict_v1 import run as predict_run

fasta_path = "swissprot_RBP.fasta"
weights = "/home/slab/ishiiayuka/M2/Decoder/t30_150M_decoder_500.pt"
protein_feat_path = "t30_150M_swissprot_RBP.pt"
output_path = "/home/slab/ishiiayuka/M2/Decoder/t30_150M__decoder_500_after_reinforce_swissprot_RBP.pt"

#逐次生成のときは以下のコードを外す
def dp_forward_parallel(model, protein_feat, out_len):
    # DataParallelでも単体でも同じ呼び方に統一
    if isinstance(model, nn.DataParallel):
        return model.module.forward_parallel(protein_feat, out_len=out_len)
    return model.forward_parallel(protein_feat, out_len=out_len)

@torch.no_grad()
#非逐次生成
def sample_tokens_from_logits(logits, mode="sample"):
    # logits: [B,L,V] → tokens: [B,L]
    probs = logits.softmax(dim=-1)
    if mode == "sample":
        B, L, V = probs.shape
        return torch.multinomial(probs.view(B*L, V), 1).view(B, L)
    else:
        return probs.argmax(dim=-1)

def logprob_from_logits(logits, tokens, pad_id):
    # logits: [B,L,V], tokens: [B,L] → logp: [B]
    logp = logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # [B,L]
    mask = (tokens != pad_id).float()
    return (logp * mask).sum(dim=-1)

def tokens_to_strings(tokens, ivocab, eos_id, pad_id, sos_id):
    # tokens: [B,L] → list[str]（<eos>で打ち切り、<pad>/<sos>は除去）
    seqs = []
    for row in tokens.tolist():
        s = []
        for t in row:
            if t == eos_id: break
            if t == pad_id or t == sos_id: continue
            s.append(ivocab[int(t)])
        seqs.append("".join(s))
    return seqs

def collate_rl(batch):
    feats = [torch.as_tensor(b[0]) for b in batch]
    protein_feat = torch.stack(feats, dim=0)          # [B, D] or [B, N, D]
    protein_seq_list = [b[2] for b in batch]          # List[str]
    return protein_feat, None, protein_seq_list, None, None

device_ids = [1,2,3]
device = f'cuda:{device_ids[0]}'
torch.cuda.set_device(device_ids[0])

#ハイパーパラメータ
baseline_mean   =  torch.tensor(0.0, device=device)          # 報酬の移動平均 (= ベースライン)
baseline_alpha  = 0.9          # どれくらいゆっくり更新するか
max_steps       = 10
grad_clip_norm  = 0.7

PRINT_SAMPLES_EVERY_STEP = 1   # 何ステップごとに表示するか（1で毎ステップ）
NUM_SHOW_PER_BATCH      = 8    # 各バッチから何本出すか
SEQ_PREVIEW_LEN         = 120  # 画面用に先頭だけ表示
REWARD_GPU_ID = 0

# --- データ準備 ---
df = pd.read_csv(config.csv_path, low_memory=False)
df["cluster_id"] = df["s1_binding_site_cluster_data_40_area"].apply(lambda x: str(x).split("_")[0])
cluster_dict = defaultdict(list)
for _, row in df.iterrows():
    cluster_dict[row["cluster_id"]].append(row["subunit_1"])

clusters = list(cluster_dict.values())
random.seed(42)
random.shuffle(clusters)
split_idx = int(0.95 * len(clusters))
train_ids = {sid for cluster in clusters[:split_idx] for sid in cluster}
dataset_train = RNADataset(config.protein_feat_path, config.csv_path, allowed_ids=train_ids)
train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn, pin_memory=True, num_workers=4, persistent_workers=True)
torch.manual_seed(42)  # PyTorch の乱数シード

# --- モデル定義 ---
model = ProteinToRNA_reinforce(input_dim=config.input_dim, num_layers=config.num_layers)
state_dict = torch.load(weights, map_location="cpu")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")  
    new_state_dict[name] = v

# 新しい state_dict を読み込む
model.load_state_dict(new_state_dict, strict=False)

# ★ 追加: 新規層の初期化（任意だが推奨）
with torch.no_grad():
    # query_embed を pos_encoder から初期化（近い表現で安定しやすい）
    if hasattr(model, "query_embed"):
        L = model.query_embed.shape[0]
        copy_len = min(L, model.pos_encoder.shape[0])
        model.query_embed[:copy_len].copy_(model.pos_encoder[:copy_len])
        if L > copy_len:
            model.query_embed[copy_len:].normal_(0.0, 0.02)  # 残りは正規分布で
    # length_head の線形層をXavierで
    if hasattr(model, "length_head"):
        for m in model.length_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

model.to(device)
model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0]) #追加
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
criterion = torch.nn.CrossEntropyLoss(ignore_index=config.rna_vocab["<pad>"])

# --- 報酬関数定義 ---
class LucaPPIReward:
    def __init__(self, model_path: str = "LucaOneTasks/",topk: Optional[int] = None):
        self.model_path = model_path
        self.topk = topk
        self.common_kwargs = dict(
            model_path = self.model_path,
            llm_truncation_seq_length = 500,
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
            topk = topk,
            gpu_id = REWARD_GPU_ID,
            emb_dir = None,
            matrix_embedding_exists = False
        )

    def __call__(self, prot_list: list[str], rna_list: list[str]) -> torch.Tensor:
        assert len(prot_list) == len(rna_list)
        sequences = [["", "", "gene", "prot", rna_list[i], prot_list[i], ""] for i in range(len(prot_list))]
        # 報酬計算は勾配不要
        with torch.no_grad():
            results = predict_run(sequences, **self.common_kwargs)

        probs = [row[4] for row in results]
        return torch.tensor(probs, dtype=torch.float32, device=device)

luca_reward = LucaPPIReward(model_path="LucaOneTasks/",topk=None)

# --- ヘルパ: トークン列 -> 文字列 ---
def tokens_to_rna_str(tokens):
    return "".join(config.rna_ivocab[int(t)] for t in tokens)

def to_rna_str_list(rna_seq):
    # rna_seq が list[int]（単一配列）の場合
    if len(rna_seq) > 0 and not isinstance(rna_seq[0], (list, tuple, torch.Tensor)):
        return [tokens_to_rna_str(rna_seq)]
    # rna_seq が list[list[int]]（複数配列）の場合
    return [tokens_to_rna_str(seq) for seq in rna_seq]

# --------------------------- 学習ループ --------------------------- #
for step in tqdm(range(max_steps),desc="step",position=0):
   
    for batch_idx, (protein_feat, tgt_seqs, protein_seq_list) in enumerate(tqdm(train_loader, desc="Samples", position=1, leave=False)):

        protein_feat = protein_feat.to(device, non_blocking=True)
        t_start = time.perf_counter()

        # ========= 1) 並列forward（勾配あり！）=========
        L = config.max_len
        logits = dp_forward_parallel(model, protein_feat, out_len=L)  # [B,L,V]

        # ========= 2) サンプリング（勾配不要）=========
        tokens = sample_tokens_from_logits(logits, mode="sample")     # [B,L]

        # ========= 3) 報酬（勾配不要）=========
        eos_id = config.rna_vocab["<eos>"]
        pad_id = config.rna_vocab["<pad>"]
        sos_id = config.rna_vocab["<sos>"]

        rna_strs = tokens_to_strings(tokens, config.rna_ivocab, eos_id, pad_id, sos_id)
        with torch.no_grad():
            R = luca_reward(protein_seq_list, rna_strs)  
            # R: [B] 想定。スカラーのときは展開
            if R.dim() == 0:
                R = R.expand(tokens.size(0))

        t_reward = time.perf_counter()

        # ========= 4) logp（勾配あり）=========
        logp_batch = logprob_from_logits(logits, tokens, pad_id=pad_id)  # [B]

        # ========= 5) REINFORCE損失 =========
        # ベースラインはEMAスカラー（勾配不要）
        with torch.no_grad():
            R_mean = R.mean()
            baseline_mean = baseline_alpha * baseline_mean + (1 - baseline_alpha) * R_mean

        advantage = (R - baseline_mean).detach()  # [B], 勾配を切る
        loss = -(advantage * logp_batch).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        t_update = time.perf_counter()

        # === 追加: 生成配列を標準出力に表示（nohupなら output.log に書かれる） ===
        if (step % PRINT_SAMPLES_EVERY_STEP == 0) :
            R_cpu = R.detach().float().cpu().tolist()
            for i, (seq, r) in enumerate(zip(rna_strs[:NUM_SHOW_PER_BATCH], R_cpu[:NUM_SHOW_PER_BATCH])):
                preview = seq[:SEQ_PREVIEW_LEN]
                tail = "..." if len(seq) > SEQ_PREVIEW_LEN else ""
                print(
                    f"[gen] step={step:03d} batch={batch_idx:04d} idx={i:02d} "
                    f"len={len(seq):4d} R={r:.4f}  {preview}{tail}",
                    flush=True  # ← 逐次反映
                )

        # ===== 保存は毎ステップでは重いので間引く =====
        if step % 5 == 0:  # 例: 5ステップごと
            torch.save(
                # DataParallelの'module.'プリフィクスを避けたいなら module 側を保存
                model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                output_path
            )