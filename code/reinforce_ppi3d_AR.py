import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from Decoder.model import ProteinToRNA
import Decoder.config as config
from Decoder.dataset import custom_collate_fn, RNADataset_AR
from Decoder.decode import sample_decode_reinforce
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict
from LucaOneTasks.src.predict_v1 import run as predict_run
import random
from collections import defaultdict
import pandas as pd

csv_path = "ppi3d.csv"
weights = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_decoder_AR_500epoch_1015.pt"
protein_feat_path = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M.pt"
output_path = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_decoder_AR_after_reinforce_ppi3d_1022.pt"

def logprob_from_logits(logits, tokens, pad_id, eos_id):
    # logits: [B,L,V], tokens: [B,L] → logp: [B]
    logp_tok = logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # [B,L]
    not_pad = (tokens != pad_id)
    eos_mask = (tokens == eos_id)
    csum = eos_mask.int().cumsum(dim=1)
    before_eos = (csum == 0)                 # eosより前（eos自身は含まず）
    first_eos  = eos_mask & (csum == 1)      # 最初のeosのみ
    include = (before_eos | first_eos) & not_pad
    return (logp_tok * include.float()).sum(dim=-1)

def tokens_to_strings(tokens, ivocab, eos_id, pad_id, sos_id):
    # tokens: [B,L] → list[str]（<eos>で打ち切り、<pad>/<sos>は除去）
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

# --- GPU設定（0〜3の4枚を使用） ---
device_ids = [1, 2]
device = f'cuda:{device_ids[0]}'
torch.cuda.set_device(device_ids[0])

# ハイパーパラメータ
baseline_mean   =  torch.tensor(0.0, device=device)  # 報酬の移動平均 (= ベースライン)
baseline_alpha  = 0.7
max_steps       = 20
grad_clip_norm  = 0.7
batch_size = 16

PRINT_SAMPLES_EVERY_STEP = 1
NUM_SHOW_PER_BATCH      = 5
SEQ_PREVIEW_LEN         = 120
REWARD_GPU_ID = device_ids[0]

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
dataset_train = RNADataset_AR(protein_feat_path, csv_path, allowed_ids=train_ids)
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
model = ProteinToRNA(input_dim=config.input_dim, num_layers=config.num_layers)
state_dict = torch.load(weights, map_location="cpu")
new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
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
            model_path=self.model_path,
            llm_truncation_seq_length=500,
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
            topk=topk,
            gpu_id=REWARD_GPU_ID,
            emb_dir=None,
            matrix_embedding_exists=False
        )
    def __call__(self, prot_list: list[str], rna_list: list[str]) -> torch.Tensor:
        assert len(prot_list) == len(rna_list)
        sequences = [["", "", "gene", "prot", rna_list[i], prot_list[i], ""] for i in range(len(prot_list))]
        with torch.no_grad():
            results = predict_run(sequences, **self.common_kwargs)
        probs = [row[4] for row in results]
        return torch.tensor(probs, dtype=torch.float32, device=device)

luca_reward = LucaPPIReward(model_path="LucaOneTasks/", topk=None)

# --- seq -> tokens（<eos> 付与して pad 埋め） ---
def pack_tokens_from_seqs(seqs, eos_id, pad_id, L):
    B = len(seqs)
    out = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        s = s[: max(0, L-1)]
        row = s + [eos_id]
        out[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return out

# --------------------------- 学習ループ --------------------------- #
eos_id = config.rna_vocab["<eos>"]
pad_id = config.rna_vocab["<pad>"]
sos_id = config.rna_vocab["<sos>"]

'''for step in tqdm(range(max_steps), desc="step", position=0):
    model.train()
    
    #追加
    optimizer.zero_grad(set_to_none=True)
    accum_steps = len(train_loader)

    for batch_idx, (protein_feat, tgt_seqs, protein_seq_list) in enumerate(
        tqdm(train_loader, desc="Samples", position=1, leave=False)
    ):
        protein_feat = protein_feat.to(device, non_blocking=True)

        # ========= 1) 逐次生成 =========
        with torch.no_grad():
            rna_idlists, _ = sample_decode_reinforce(
            model, protein_feat,
            max_len=config.max_len,
            num_samples=1,
            top_k=config.top_k,
            temperature=1.0,
            min_len=config.min_len
        )
        tokens = pack_tokens_from_seqs(rna_idlists, eos_id, pad_id, L=config.max_len)

        # ========= 2) 報酬 =========
        rna_strs = tokens_to_strings(tokens, config.rna_ivocab, eos_id, pad_id, sos_id)
        with torch.no_grad():
            R = luca_reward(protein_seq_list, rna_strs)
            if R.dim() == 0:
                R = R.expand(tokens.size(0))

        # ========= 3) logp =========
        prefix = torch.full((tokens.size(0), 1), sos_id, device=device, dtype=torch.long)
        L_TF = min(tokens.size(1), config.max_len - 1)
        inp  = torch.cat([prefix, tokens[:, :L_TF]], dim=1)   # [B, 1+L_TF]
        logits = model(protein_feat, inp)[:, 1:, :] 
        tokens_tf = tokens[:, :logits.size(1)]                # [B, L_TF]

        logp_batch = logprob_from_logits(logits, tokens_tf, pad_id=pad_id, eos_id=eos_id)

        # ========= 4) REINFORCE損失 =========
        with torch.no_grad():
            R_mean = R.mean()
            baseline_mean = baseline_alpha * baseline_mean + (1 - baseline_alpha) * R_mean

        advantage = (R - baseline_mean).detach()
        loss = -(advantage * logp_batch).mean()

        (loss / accum_steps).backward()

        # ログ出力
        if (step % PRINT_SAMPLES_EVERY_STEP == 0):
            R_cpu = R.detach().float().cpu().tolist()
            for i, (seq, r) in enumerate(zip(rna_strs[:NUM_SHOW_PER_BATCH], R_cpu[:NUM_SHOW_PER_BATCH])):
                preview = seq[:SEQ_PREVIEW_LEN]
                tail = "..." if len(seq) > SEQ_PREVIEW_LEN else ""
                print(f"[gen] step={step:03d} batch={batch_idx:04d} idx={i:02d} len={len(seq):4d} R={r:.4f}  {preview}{tail}", flush=True)

    clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()

    if (step + 1) % 5 == 0:
        save_path = f"{output_path}_{step+1}step.pt"
        torch.save(
            model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            save_path
        )'''

# 事前にIDを1回だけ取得（AR版の語彙を想定）
eos_id = config.rna_vocab["<eos>"]
pad_id = config.rna_vocab["<pad>"]
sos_id = config.rna_vocab["<sos>"]

for batch_idx, (protein_feat, tgt_seqs, protein_seq_list) in enumerate(
    tqdm(train_loader, desc="Batches", position=0)
):
    protein_feat = protein_feat.to(device, non_blocking=True)

    # バッチ固定で max_steps 回の更新
    for step in tqdm(range(max_steps), desc=f"Steps (batch {batch_idx})", position=1, leave=False):
        model.train()

        # ステップごとに勾配をクリア（累積しない）
        optimizer.zero_grad(set_to_none=True)

        # ========= 1) 逐次生成（探索は勾配不要）=========
        with torch.no_grad():
            rna_idlists, _ = sample_decode_reinforce(
                model, protein_feat,
                max_len=config.max_len,
                num_samples=1,
                top_k=config.top_k,
                temperature=1.0,
                min_len=config.min_len
            )
        tokens = pack_tokens_from_seqs(rna_idlists, eos_id, pad_id, L=config.max_len)

        # ========= 2) 報酬 =========
        rna_strs = tokens_to_strings(tokens, config.rna_ivocab, eos_id, pad_id, sos_id)
        with torch.no_grad():
            R = luca_reward(protein_seq_list, rna_strs)
            if R.dim() == 0:
                R = R.expand(tokens.size(0))

        # ========= 3) 教師ありパスで logp =========
        prefix = torch.full((tokens.size(0), 1), sos_id, device=device, dtype=torch.long)
        L_TF = min(tokens.size(1), config.max_len - 1)
        inp  = torch.scat([prefix, tokens[:, :L_TF]], dim=1)  # [B, 1+L_TF]
        logits = model(protein_feat, inp)[:, 1:, :]          # [B, L_TF, V]
        tokens_tf = tokens[:, :logits.size(1)]               # [B, L_TF]

        logp_batch = logprob_from_logits(logits, tokens_tf, pad_id=pad_id, eos_id=eos_id)

        # --- 3.5) エントロピー正則化（長さ正規化して平均）---
        probs = logits.softmax(dim=-1)                 # [B, L_TF, V]
        logp_all = (probs + 1e-8).log()
        H_tok = -(probs * logp_all).sum(dim=-1)        # [B, L_TF]

        eos_mask = (tokens_tf == eos_id)
        not_pad_mask = (tokens_tf != pad_id)
        csum = eos_mask.int().cumsum(dim=1)
        before_eos = (csum == 0)
        first_eos  = eos_mask & (csum == 1)
        include = (before_eos | first_eos) & not_pad_mask
        len_norm = include.float().sum(1).clamp_min(1.0)

        H_seq = (H_tok * include.float()).sum(1) / len_norm
        entropy_bonus = H_seq.mean()
        beta_entropy = 0.02  # ← 0.01〜0.05 で調整

        # ========= 4) REINFORCE損失 =========
        with torch.no_grad():
            R_mean = R.mean()
            baseline_mean = baseline_alpha * baseline_mean + (1 - baseline_alpha) * R_mean

        advantage = (R - baseline_mean).detach()
        loss = -(advantage * logp_batch).mean()

        #追加
        loss = loss - beta_entropy * entropy_bonus

        # 逆伝播＆更新（このステップ分だけで更新）
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        # ログ出力（任意）
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

