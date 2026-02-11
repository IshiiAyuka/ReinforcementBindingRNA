import os
import gzip
import glob
import torch
import torch.nn as nn
import esm
from tqdm import tqdm
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1
import pandas as pd

layer = 30

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ESMモデルのラッパー
class ESMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens):
        return self.model(tokens=tokens, repr_layers=[layer])

# ESM2モデルの読み込み
model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
model = ESMWrapper(model).to(device)
batch_converter = alphabet.get_batch_converter()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.eval()

# 特徴量抽出
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

# === 特徴量抽出実行 ===
protein_features = {}
df = pd.read_csv("RNAcompete.csv")

for _, row in tqdm(df.iterrows(), total=len(df), desc="特徴量抽出中 (CSV)"):
    uid = str(row["file_name"]).strip()       # 例: "3af6_A"
    seq = str(row["sequence"]).strip().upper()

    if not uid or not seq:
        print(f"Skip（欠損）: uid={uid}, len={len(seq) if seq else 0}")
        continue

    if uid in protein_features:
        continue

    L = len(seq)
    if L > 1022:
        print(f"Skip {uid}（長さ{L}）: 配列長が条件外")
        continue

    try:
        rep = extract_features(uid, seq)
        protein_features[uid] = rep
        print(f"成功: {uid}（長さ{len(seq)}）")
    except Exception as e:
        print(f"Error processing {uid}: {e}")
        torch.cuda.empty_cache()
        continue

# 保存
torch.save(protein_features, "t30_150M_RNAcompete_3D.pt")
print("特徴量を保存しました。")
