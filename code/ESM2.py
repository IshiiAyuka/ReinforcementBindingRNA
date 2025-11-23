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

# CIFからChainごとにアミノ酸配列を抽出
def extract_aa_seqs_from_cif(cif_path):
    parser = MMCIFParser(QUIET=True)
    aa_seqs = {}
    try:
        with gzip.open(cif_path, "rt") as handle:
            structure = parser.get_structure("complex", handle)
            model_obj = next(structure.get_models())

            for chain in model_obj:
                seq = ""
                for residue in chain:
                    resname = residue.get_resname()
                    try:
                        aa = seq1(resname)
                        seq += aa
                    except Exception:
                        continue
                if len(seq) > 0:
                    chain_id = chain.id
                    aa_seqs[chain_id] = seq
    except Exception as e:
        print(f"エラー: {cif_path} - {e}")
    return aa_seqs

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
cif_files = glob.glob("./filtered_data/*.cif.gz")  # パスは必要に応じて変更

for cif_path in tqdm(cif_files, desc="特徴量抽出中"):
    complex_id = os.path.basename(cif_path).replace(".cif.gz", "")
    aa_seqs = extract_aa_seqs_from_cif(cif_path)

    for chain_id, seq in aa_seqs.items():
        name = f"{complex_id}_{chain_id}"
        if len(seq)<10 or len(seq) > 1000:
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
torch.save(protein_features, "t30_150M_deepclip.pt")
print("特徴量を保存しました。")