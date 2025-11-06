import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from Decoder.model import ProteinToRNA_NAR
import Decoder.config as config
from Decoder.dataset import custom_collate_fn, RNADataset_NAR
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Optional
from collections import OrderedDict
from LucaOneTasks.src.predict_v1 import run as predict_run
import random
from collections import defaultdict
import pandas as pd

csv_path = "ppi3d.csv"
weights = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_decoder_NAR_500epoch.pt"
protein_feat_path = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M.pt"
output_path = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_decoder_NAR_after_reinforce_ppi3d_1031.pt"

# --- logp: <eos> を含み、それ以降は無視（<pad>も無視） ---  
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
    return num / den  # ← 長さ正規化（平均）

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

# --- GPU設定（0〜3の4枚を使用） ---
device_ids = [0, 1, 2]
device = f'cuda:{device_ids[0]}'
torch.cuda.set_device(device_ids[0])

# ハイパーパラメータ
baseline_mean   =  torch.tensor(0.0, device=device)          # 報酬の移動平均 (= ベースライン)
baseline_alpha  = 0.7
max_steps       = 10
grad_clip_norm  = 0.7
MIN_LEN = 10
batch_size = 16

PRINT_SAMPLES_EVERY_STEP = 1
NUM_SHOW_PER_BATCH      = 5
SEQ_PREVIEW_LEN         = 120
REWARD_GPU_ID = device_ids[0] 

# --- オフターゲット抑制（シンプル） ---
OFFTARGET_LAMBDA = 0.7   # R_eff = R - λ * R_off の λ（0.3~0.7 で探索推奨）
OFFTARGET_MARGIN = 0.05 

# --- データ準備 ---
df = pd.read_csv(csv_path, low_memory=False)
df["cluster_id"] = df["s1_binding_site_cluster_data_40_area"].apply(lambda x: str(x).split("_")[0])
cluster_dict = defaultdict(list)
for _, row in df.iterrows():
    cluster_dict[row["cluster_id"]].append(row["subunit_1"])

clusters = list(cluster_dict.values())
random.seed(42)
random.shuffle(clusters)
split_idx = int(0.95 * len(clusters))
train_ids = {sid for cluster in clusters[:split_idx] for sid in cluster}
dataset_train = RNADataset_NAR(protein_feat_path, csv_path, allowed_ids=train_ids)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, pin_memory=True, num_workers=4, persistent_workers=True)
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

# --- 報酬関数定義 ---
class LucaPPIReward:
    def __init__(self, model_path: str = "LucaOneTasks/", topk: Optional[int] = None):
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
        with torch.no_grad():
            results = predict_run(sequences, **self.common_kwargs)
        probs = [row[4] for row in results]
        return torch.tensor(probs, dtype=torch.float32, device=device)

luca_reward = LucaPPIReward(model_path="LucaOneTasks/", topk=None)

# --- ヘルパ: トークン列 -> 文字列 ---
def tokens_to_rna_str(tokens):
    return "".join(config.rna_ivocab_NAR[int(t)] for t in tokens)

def to_rna_str_list(rna_seq):
    if len(rna_seq) > 0 and not isinstance(rna_seq[0], (list, tuple, torch.Tensor)):
        return [tokens_to_rna_str(rna_seq)]
    return [tokens_to_rna_str(seq) for seq in rna_seq]

# --------------------------- 学習ループ（エポックごとに更新） ---------------------------
'''for epoch in tqdm(range(max_steps), desc="epoch", position=0):
    model.train()

    # ← エポック開始時に一度だけゼロクリア
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (protein_feat, tgt_seqs, protein_seq_list) in enumerate(
        tqdm(train_loader, desc="Samples", position=1, leave=False)
    ):
        protein_feat = protein_feat.to(device, non_blocking=True)

        eos_id = config.rna_vocab_NAR["<eos>"]
        pad_id = config.rna_vocab_NAR["<pad>"]
        sos_id = config.rna_vocab_NAR["<sos>"]
        mask_id = config.rna_vocab_NAR["<MASK>"]

        # ========= 1) 並列forward =========
        logits = model.module.forward_parallel(protein_feat, out_len=config.max_len)  # [B,L,V]

        # ========= 2) サンプリング（禁止トークンをban） =========
        logits[..., pad_id]  = -1e9
        logits[..., sos_id]  = -1e9
        logits[..., mask_id] = -1e9
        _min_len = min(MIN_LEN, logits.size(1))
        logits[:, :_min_len, eos_id] = -1e9

        probs = logits.softmax(dim=-1)
        B, L, V = probs.shape
        tokens = torch.multinomial(probs.reshape(B * L, V), 1).reshape(B, L)

        # ========= 3) 報酬 =========
        rna_strs = tokens_to_strings(tokens, config.rna_ivocab_NAR, eos_id, pad_id, sos_id)
        with torch.no_grad():
            R = luca_reward(protein_seq_list, rna_strs)
            if R.dim() == 0:
                R = R.expand(tokens.size(0))

        # ========= 4) logp（<eos>以降は無視、<eos>は含める） =========
        logp_batch = logprob_from_logits(logits, tokens, pad_id=pad_id, eos_id=eos_id)

        # ========= 5) REINFORCE損失（勾配は蓄積、ここではstepしない） =========
        with torch.no_grad():
            R_mean = R.mean()
            baseline_mean = baseline_alpha * baseline_mean + (1 - baseline_alpha) * R_mean

        advantage = (R - baseline_mean).detach()
        loss = -(advantage * logp_batch).mean()

        # ← バッチごとにbackwardだけ実行（勾配は蓄積される）
        loss.backward()

        # ログ出力（エポック番号で表示）
        if (epoch % PRINT_SAMPLES_EVERY_STEP == 0):
            R_cpu = R.detach().float().cpu().tolist()
            for i, (seq, r) in enumerate(zip(rna_strs[:NUM_SHOW_PER_BATCH], R_cpu[:NUM_SHOW_PER_BATCH])):
                preview = seq[:SEQ_PREVIEW_LEN]
                tail = "..." if len(seq) > SEQ_PREVIEW_LEN else ""
                print(f"[gen] epoch={epoch:03d} batch={batch_idx:04d} idx={i:02d} len={len(seq):4d} R={r:.4f}  {preview}{tail}", flush=True)

    clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    # 保存
    if (epoch + 1) % 5 == 0:
        save_path = f"{output_path}_{epoch+1}step.pt"
        torch.save(
            model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            save_path
        )
'''

# ====== バッチごとに全ステップを完了してから次のバッチへ ======
for batch_idx, (protein_feat, tgt_seqs, protein_seq_list) in enumerate(
    tqdm(train_loader, desc="Batches", position=0)
):
    protein_feat = protein_feat.to(device, non_blocking=True)

    # 追加: このバッチ内での前後ターゲット（端は隣を複製）
    B_batch = len(protein_seq_list)
    if B_batch > 1:
        prev_idx = [i-1 if i > 0 else 1 for i in range(B_batch)]
        next_idx = [i+1 if i < B_batch-1 else B_batch-2 for i in range(B_batch)]
    else:
        prev_idx = [0]; next_idx = [0] 

    # バッチ固定で max_steps 回の更新を回す
    for step in tqdm(range(max_steps), desc=f"Steps (batch {batch_idx})", position=1, leave=False):
        model.train()

        optimizer.zero_grad(set_to_none=True)

        eos_id  = config.rna_vocab_NAR["<eos>"]
        pad_id  = config.rna_vocab_NAR["<pad>"]
        sos_id  = config.rna_vocab_NAR["<sos>"]
        mask_id = config.rna_vocab_NAR["<MASK>"]

        # ========= 1) 並列forward =========
        logits = model.module.forward_parallel(protein_feat, out_len=config.max_len)  # [B,L,V]

        # ========= 2) サンプリング（禁止トークンをban） =========
        logits[..., pad_id]  = -1e9
        logits[..., sos_id]  = -1e9
        logits[..., mask_id] = -1e9
        _min_len = min(MIN_LEN, logits.size(1))
        logits[:, :_min_len, eos_id] = -1e9

        probs = logits.softmax(dim=-1)
        B, L, V = probs.shape
        tokens = torch.multinomial(probs.reshape(B * L, V), 1).reshape(B, L)

        # ========= 2.5) エントロピー正則化（長さ正規化で） =========
        # ＊no_gradにしない：この項の勾配を流すため
        logp_all = (probs + 1e-8).log()                    # [B,L,V]
        H_tok = -(probs * logp_all).sum(dim=-1)            # [B,L] 各位置のエントロピー

        eos_mask = (tokens == eos_id)
        csum = eos_mask.int().cumsum(dim=1)
        before_eos = (csum == 0)
        first_eos  = eos_mask & (csum == 1)
        include = (before_eos | first_eos)                 # <eos>まで＋最初の<eos>を含む
        len_norm = include.float().sum(1).clamp_min(1.0)   # [B]

        H_seq = (H_tok * include.float()).sum(1) / len_norm   # 各系列の平均エントロピー
        entropy_bonus = H_seq.mean()                          # バッチ平均
        beta_entropy = 0.02   

        # ========= 3) 報酬 =========
        rna_strs = tokens_to_strings(tokens, config.rna_ivocab_NAR, eos_id, pad_id, sos_id)
        with torch.no_grad():
            R = luca_reward(protein_seq_list, rna_strs)
            if R.dim() == 0:
                R = R.expand(tokens.size(0))
        
        # ========= 3.5) オフターゲット（前/後のタンパク質に対するスコア）=========
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
                scores_off = luca_reward(prot_off, rna_off)  # 長さ B*(B-1)
            R_off = scores_off.reshape(B_batch, B_batch - 1).max(dim=1).values  # [B]
        else:
            R_off = torch.zeros_like(R)

        R_eff = R - OFFTARGET_LAMBDA * R_off

        # ========= 4) logp（<eos>以降は無視、<eos>は含める） =========
        logp_batch = logprob_from_logits(logits, tokens, pad_id=pad_id, eos_id=eos_id)

        # ========= 5) REINFORCE損失 =========
        with torch.no_grad():
            R_mean = R_eff.mean()
            baseline_mean = baseline_alpha * baseline_mean + (1 - baseline_alpha) * R_mean

        advantage = (R_eff - baseline_mean).detach()
        loss = -(advantage * logp_batch).mean()

        #追加
        loss = loss - beta_entropy * entropy_bonus

        # 逆伝播＆更新（このステップ分だけで更新）
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        # ログ出力（ステップ番号で表示）
        if (step % PRINT_SAMPLES_EVERY_STEP == 0):
            R_cpu = R.detach().float().cpu().tolist()
            for i, (seq, r) in enumerate(zip(rna_strs[:NUM_SHOW_PER_BATCH], R_cpu[:NUM_SHOW_PER_BATCH])):
                preview = seq[:SEQ_PREVIEW_LEN]
                tail = "..." if len(seq) > SEQ_PREVIEW_LEN else ""
                print(f"[gen] batch={batch_idx:04d} step={step:06d} idx={i:02d} len={len(seq):4d} R={r:.4f}  {preview}{tail}", flush=True)

final_path = f"{output_path}"
torch.save(
    model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
    final_path
)
print(f"[save] Final weights saved to: {final_path}", flush=True)
