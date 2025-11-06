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
    batch_labels, batch_strs, batch_tokens = batch_converter([(name, seq)])
    batch_tokens = batch_tokens.to(device)
    results = model(batch_tokens)
    token_representations = results["representations"][layer]
    tokens_len = (batch_tokens != alphabet.padding_idx).sum(1)[0]
    rep = token_representations[0, 1:tokens_len - 1].mean(0).cpu()
    del batch_tokens, results, token_representations
    torch.cuda.empty_cache()
    return rep

# === 特徴量抽出実行 ===
protein_features = {}
# === CSVから配列を読み込み・特徴量抽出 ===
CSV_PATH = "/home/slab/ishiiayuka/M2/RNAcompete.csv"  
ID_COLS  = ["file_name", "protein_name"]  # IDをこれらから組み立て
SEQ_COL  = "sequence"
# CSV読み込み
df = pd.read_csv(CSV_PATH)

for _, row in tqdm(df.iterrows(), total=len(df), desc="特徴量抽出中 (CSV)"):
    pname = str(row.get("protein_name", "")).strip()
    fname = str(row.get("file_name", "")).strip()

    parts = [x for x in [fname, pname] if x]
    name = "_".join(parts) if parts else "unknown"

    seq = str(row[SEQ_COL]).strip().upper()

    if not name or not seq:
        print(f"Skip（欠損）: id={name}, len={len(seq) if seq else 0}")
        continue
    if len(seq) < 10 or len(seq) > 1000:
        print(f"Skip {name}（長さ{len(seq)}）: 配列長が条件外")
        continue

    try:
        rep = extract_features(name, seq)
        protein_features[name] = rep
        print(f"成功: {name}（長さ{len(seq)}）")
    except Exception as e:
        print(f"Error processing {name}: {e}")
        torch.cuda.empty_cache()
        continue

# 保存
torch.save(protein_features, "t30_150M_RNAcompete.pt")
print("特徴量を保存しました。")