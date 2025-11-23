import os
import torch
import torch.nn as nn
import esm
from tqdm import tqdm
from Bio import SeqIO

# ===== 設定 =====
fasta_path = "swissprot_RBP.fasta"
out_path   = "t30_150M_swissprot_RBP_3D.pt"
layer = 30            

# ===== デバイス / モデル =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[device]", device, "| #GPUs =", torch.cuda.device_count())

class ESMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens):
        return self.model(tokens=tokens, repr_layers=[layer])

model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model = ESMWrapper(model).to(device)
batch_converter = alphabet.get_batch_converter()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.eval()

# ===== バッチ特徴量抽出 =====
@torch.no_grad()
def extract_features(name, seq):
    _, _, batch_tokens = batch_converter([(name, seq)])
    batch_tokens = batch_tokens.to(device)
    results = model(batch_tokens)
    token_representations = results["representations"][layer]
    tokens_len = (batch_tokens != alphabet.padding_idx).sum(1)[0].item()
    rep = token_representations[0, 1:tokens_len - 1].cpu()
    del batch_tokens, results, token_representations
    torch.cuda.empty_cache()
    return rep

# ===== 実行 =====
protein_features = {}  # { "A0A075QQ08": tensor[D], ... }
records = list(SeqIO.parse(fasta_path, "fasta"))

for record in tqdm(records, desc="特徴量抽出中 (FASTA)"):
    # 例: record.id = "sp|A0A075QQ08|IF4E1_TOBAC"
    raw_id = str(record.id).strip()
    parts = raw_id.split("|")

    # sp|ACC|NAME の形式なら ACC だけを使う
    if len(parts) >= 3 and parts[0] in ("sp", "tr"):
        uid = parts[1]        # 例: "A0A075QQ08"
    else:
        uid = raw_id          # それ以外はそのまま

    seq = str(record.seq).strip().upper()

    if not uid or not seq:
        print(f"Skip（欠損）: uid={uid}, len={len(seq) if seq else 0}")
        continue

    # すでに同じ ID を処理済みならスキップ
    if uid in protein_features:
        continue

    L = len(seq)
    if L > 1000:
        print(f"Skip {uid}（長さ{L}）: 配列長が条件外")
        continue

    try:
        rep = extract_features(uid, seq)  # [L, 640]
        protein_features[uid] = rep
        print(f"成功: {uid}（長さ{L}, shape={tuple(rep.shape)}）")
    except Exception as e:
        print(f"Error processing {uid}: {e}")
        torch.cuda.empty_cache()
        continue

# ===== 保存 & 簡単な確認 =====
torch.save(protein_features, out_path)
print(f"特徴量を {out_path} に保存しました。")

loaded = torch.load(out_path, map_location="cpu")
for i, (k, v) in enumerate(loaded.items()):
    print(f"{i+1}. {k}: shape={tuple(v.shape)}, dim={v.dim()}")
    if i >= 4:
        break