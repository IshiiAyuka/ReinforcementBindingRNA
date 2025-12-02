import torch
import torch.nn as nn
from dataset import RNADataset_deepclip_AR, custom_collate_fn_deepclip_AR
from decode import sample_decode_multi_AR
from model import ProteinToRNA
import config
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict

CKPT_PATH = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_decoder_AR_reinforce_1128.pt"
protein_feat_path = "/home/slab/ishiiayuka/M2/t30_150M_RNAcompete_3D.pt"
csv_path = "/home/slab/ishiiayuka/M2/RNAcompete.csv"
output_path = "/home/slab/ishiiayuka/M2/generated_rna_RNCMPT_t30_150M_AR_1202.csv"
num_samples = 100

if __name__ == "__main__":
    # --- データ準備 ---

    df = pd.read_csv(csv_path, low_memory=False)

    dataset_train = RNADataset_deepclip_AR(protein_feat_path, csv_path)
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn_deepclip_AR)

    print(f"Trainデータ数: {len(dataset_train)}")

    # --- モデル定義 ---
    model = ProteinToRNA(input_dim=config.input_dim, num_layers=config.num_layers)
    model = nn.DataParallel(model)
    model = model.to(config.device)

    # --- 学習済み重みロード（推論のみ） ---
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    if any(k.startswith("module.") for k in state_dict.keys()):
        model.load_state_dict(state_dict, strict=True)
    else:
        model.module.load_state_dict(state_dict, strict=True)

    model.eval()

    id2tok = {v: k for k, v in config.rna_vocab.items()}

    # --- 生成してCSVへ ---
    rows = []
    with torch.inference_mode():
        for batch in tqdm(train_loader):
            # custom_collate_fn_deepclip_AR が (protein_batch, prot_seqs, file_names) を返す想定
            protein_batch, prot_seqs, file_names = batch

            out_all = sample_decode_multi_AR(
                model,
                protein_batch,                 # [B, S, D]
                max_len=config.max_len,
                num_samples=num_samples,       # 1タンパク質あたり100本
                top_k=config.top_k,
                temperature=config.temp,
            )
            for i, fn in enumerate(file_names):
                base_id = str(fn).strip()
                if base_id.endswith(".pkl"):
                    base_id = base_id[:-4]  # ".pkl" を除去

                for seq_ids in out_all[i]:
                    if not seq_ids:
                        continue
                    rna_seq = "".join(id2tok[t] for t in seq_ids)
                    rows.append({"id": base_id, "sequence": rna_seq})

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved: {output_path} ")
