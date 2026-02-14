import os
import torch
import torch.nn as nn
import esm
from tqdm import tqdm
from Bio import SeqIO
import argparse

# ===== 設定 =====
layer = 30

MAX_LEN = 1022  

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

# ===== 引数 =====
parser = argparse.ArgumentParser(description="Extract ESM2 features from a FASTA file.")
parser.add_argument("input_fasta", help="入力FASTAファイルのパス")
parser.add_argument("output_pt", help="出力PTファイルのパス")
args = parser.parse_args()

# 入出力パス
fasta_path = args.input_fasta
out_path = args.output_pt

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
protein_features = {}  # { "A0A075QQ08": tensor[L, 640], ... }
records = list(SeqIO.parse(fasta_path, "fasta"))

# カウント
n_total = 0
n_skip_missing = 0
n_skip_dup = 0
n_skip_long = 0
n_error = 0
n_ok = 0

for record in tqdm(records, desc="特徴量抽出中 (FASTA)"):
    n_total += 1

    raw_id = str(record.id).strip()
    parts = raw_id.split("|")

    if len(parts) >= 3 and parts[0] in ("sp", "tr"):
        uid = parts[1]
    else:
        uid = raw_id

    seq = str(record.seq).strip().upper()

    if not uid or not seq:
        print(f"Skip（欠損）: uid={uid}, len={len(seq) if seq else 0}")
        n_skip_missing += 1
        continue

    if uid in protein_features:
        n_skip_dup += 1
        continue

    L = len(seq)
    if L > MAX_LEN:  
        print(f"Skip {uid}（長さ{L}）: 配列長が条件外（>=1023）")
        n_skip_long += 1
        continue

    try:
        rep = extract_features(uid, seq)  # [L, 640]
        protein_features[uid] = rep
        n_ok += 1
        print(f"成功: {uid}（長さ{L}, shape={tuple(rep.shape)}）")
    except Exception as e:
        n_error += 1
        print(f"Error processing {uid}: {e}")
        torch.cuda.empty_cache()
        continue

torch.save(protein_features, out_path)
print(f"特徴量を {out_path} に保存しました。")
