import os
import torch
import torch.nn as nn
import esm
from tqdm import tqdm

# ===== 設定 =====
fasta_path = "swissprot_all.fasta"
out_path   = "t30_150M_swissprot_all.pt"
layer = 30                              # esm2_t30_150M_UR50D の最終層
BATCH_SIZE = 16                         # GPUメモリに合わせて調整

# ===== デバイス / モデル =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[device]", device, "| #GPUs =", torch.cuda.device_count())

base_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()

class ESMWrapper(nn.Module):
    def __init__(self, model, layer: int):
        super().__init__()
        self.model = model
        self.layer = layer
    def forward(self, tokens):
        out = self.model(tokens=tokens, repr_layers=[self.layer])
        return out["representations"][self.layer]  # [B,L,D]

model = ESMWrapper(base_model, layer).to(device)
batch_converter = alphabet.get_batch_converter()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.eval()

# ===== FASTA 読み込み =====
def read_fasta(path: str):
    recs = []
    with open(path, "r") as f:
        header = None
        seq = []
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                if header is not None:
                    recs.append({"header": header, "seq": "".join(seq).replace(" ", "").upper()})
                header = s[1:].split()[0]  # 先頭トークン（例: sp|A0A075QQ08|IF4E1_TOBAC）
                seq = []
            else:
                seq.append(s)
        if header is not None:
            recs.append({"header": header, "seq": "".join(seq).replace(" ", "").upper()})
    return recs

def header_to_accession(header: str) -> str:
    """UniProt系 'sp|ACC|ID' 形式ならACCを返す。そうでなければheader全体を返す。"""
    parts = header.split("|")
    return parts[1] if len(parts) >= 2 else header

# ===== バッチ特徴量抽出 =====
@torch.no_grad()
def extract_batch(names, seqs):
    # ESMのバッチ変換（CPU上でOK）
    _, _, tokens = batch_converter(list(zip(names, seqs)))   # [B,L]
    # DataParallel使用時はCPUのまま渡すと自動で各GPUへscatterされる
    if not isinstance(model, nn.DataParallel):
        tokens = tokens.to(device)

    reps = model(tokens)  # [B,L,D] （DPのときは自動で集約されdevice[0]上に載る）
    # 可変長処理のため、CPU側tokensで実長を計算（BOS/EOSを除外）
    lengths = (tokens != alphabet.padding_idx).sum(dim=1).tolist()  # 各サンプルの有効長

    feats = []
    for i, L in enumerate(lengths):
        if L < 3:
            # 実質的に配列が無いケース（BOS/EOSのみ）をスキップ
            feats.append(None)
            continue
        # 1 .. L-2 が実アミノ酸領域（BOS=0, EOS=L-1）
        vec = reps[i, 1:L-1, :].mean(dim=0).detach().cpu()  # [D]
        feats.append(vec)
    return feats

# ===== 実行 =====
records = read_fasta(fasta_path)
protein_features = {}  # { "A0A075QQ08": tensor[D], ... }

# 長さフィルタ
filtered = [(r["header"], r["seq"]) for r in records if 10 <= len(r["seq"]) <= 1000]

for i in tqdm(range(0, len(filtered), BATCH_SIZE), desc="特徴量抽出中"):
    chunk = filtered[i:i+BATCH_SIZE]
    names = [h for h, _ in chunk]
    seqs  = [s for _, s in chunk]

    feats = extract_batch(names, seqs)
    for h, s, f in zip(names, seqs, feats):
        if f is None:
            continue
        acc = header_to_accession(h)     # ← ここでキーをアクセッションだけに統一！
        protein_features[acc] = f

# 保存（{アクセッション: Tensor[D]}）
torch.save(protein_features, out_path)
print(f"[done] 保存: {out_path} | 件数={len(protein_features)} | 次元={next(iter(protein_features.values())).shape[0] if protein_features else 'N/A'}")