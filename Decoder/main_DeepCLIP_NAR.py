import torch
import torch.nn as nn
from dataset import DeepCLIPProteinDataset
from decode import sample_decode_multi
from model import ProteinToRNA_NAR
import config
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict

CKPT_PATH = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_decoder_NAR_after_reinforce_ppi3d_1031.pt"
protein_feat_path = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_RNAcompete.pt"
csv_path = "/home/slab/ishiiayuka/M2/RNAcompete.csv"
output_path = "/home/slab/ishiiayuka/M2/generated_rna_RNCMPT_t30_150M_NAR_1102.csv"
num_samples = 100

# ---- state_dict ロード（module. 剥がし + drop_prefixes 対応、簡潔版）----
def load_checkpoint_flex(model, ckpt_path, device, drop_prefixes=("length_head.", "query_embed")):
    raw = torch.load(ckpt_path, map_location=device)
    sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if all(isinstance(k, str) and k.startswith("module.") for k in sd.keys()):
        sd = OrderedDict((k[len("module."):], v) for k, v in sd.items())
    if drop_prefixes:
        sd = OrderedDict((k, v) for k, v in sd.items() if not any(k.startswith(p) for p in drop_prefixes))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

# ---- 推論用 collate（feat と id のみ）----
def deepclip_collate_for_predict(batch):
    feats = torch.stack([b[0] for b in batch], dim=0)  # [B, D] or [B, S, D]
    keys  = [b[1] for b in batch]
    return feats, keys

# --- 実行部分 ---
dataset_test = DeepCLIPProteinDataset(feat_pt_path=protein_feat_path, csv_path=csv_path,id_priority=("uniprot_id", "protein_name", "file_name"))
print(f"Testデータ数: {len(dataset_test)}")
test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, collate_fn=deepclip_collate_for_predict)

model = ProteinToRNA_NAR(input_dim=config.input_dim, num_layers=config.num_layers).to(config.device)
load_checkpoint_flex(model, CKPT_PATH, config.device)   # ← 先にロード
model = nn.DataParallel(model).to(config.device)
model.eval()
print("モデルの重みを読み込みました。")

# ---- トークン列 → RNA 文字列（<eos>打切り、<pad>/<sos>/<MASK>除外）----
PAD  = config.rna_vocab_NAR["<pad>"]
SOS  = config.rna_vocab_NAR["<sos>"]
EOS  = config.rna_vocab_NAR["<eos>"]
MASK = config.rna_vocab_NAR["<MASK>"]

def ids_to_clean_rna(seq_ids):
    out = []
    for t in seq_ids:  # list / tensor どちらでもOK
        tid = int(t)
        # EOS が来ても、十分な長さに達するまでは無視
        if tid == EOS:
            if len(out) >= config.min_len:
                break
            else:
                continue
        if tid in (PAD, SOS, MASK):
            continue
        tok = config.rna_ivocab_NAR.get(tid)
        if tok in ("A", "U", "C", "G"):
            out.append(tok)
    return "".join(out)

# --- 生成＆保存 ---
# 複数配列
results = []
with torch.no_grad():
    for protein_batch, uniprot_ids in tqdm(test_loader, desc="RNA配列生成中"):
        protein_batch = protein_batch.to(config.device, non_blocking=True)

        seq_ids_batch = sample_decode_multi(
            model, protein_batch,
            max_len=config.max_len,
            num_samples=num_samples,
            top_k=config.top_k,
            temperature=1.0
        )

        for idx, seq_ids in zip(uniprot_ids, seq_ids_batch):
            short_id = str(idx).split("_", 1)[0]
            for one in seq_ids:
                    seq = ids_to_clean_rna(one)
                    results.append({"id": short_id, "sequence": seq})

# --- CSVとして保存（DeepCLIP互換） ---
df_out = pd.DataFrame(results)
df_out.to_csv(output_path, index=False)
print("全候補をcsvファイルに保存しました。")