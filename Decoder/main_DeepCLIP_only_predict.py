import torch
import torch.nn as nn
from dataset import RNAOnlyDataset
from decode import sample_decode 
from model import ProteinToRNA
import config
import pandas as pd
from tqdm import tqdm

# --- 実行部分 ---
dataset_test = RNAOnlyDataset("weights/t6_8M_deepclip_RNCMPT.pt")
print(f"Testデータ数: {len(dataset_test)}")

model = ProteinToRNA(input_dim=config.input_dim, num_layers=config.num_layers)
model = nn.DataParallel(model)
model = model.to(config.device)
model.load_state_dict(torch.load("t6_8M_decoder.pt", map_location=config.device))
model.eval()
print("モデルの重みを読み込みました。")

# --- 生成＆保存 ---
results = []
for protein_feat, uniprot_id in tqdm(dataset_test, desc="RNA配列生成中"):
    feats = protein_feat.to(config.device)
    seq_ids_list = sample_decode(model, feats, max_len=config.max_len, num_samples=config.num_samples)
    for idx, seq_ids in enumerate(seq_ids_list, 1):
        seq = "".join([config.rna_ivocab[i] for i in seq_ids])
        print(f"{uniprot_id} - シーケンス {idx}: {seq}")
        results.append({"id": uniprot_id, "sequence": seq, "rank": idx})

# --- CSVとして保存（DeepCLIP互換） ---
df_out = pd.DataFrame(results)
df_out.to_csv("generated_rna_trimmed_12_75.csv", index=False)
print("全候補をcsvファイルに保存しました。")