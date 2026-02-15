import torch
import torch.nn as nn
import csv
import sys
import argparse
from decode import sample_decode_multi_AR
from model import ProteinToRNA
import config
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from collections import defaultdict
import random

split_seed = 42
num_samples = 1


class RNADataset_AR(Dataset):
    def __init__(self, protein_feat_file, csv_path, allowed_ids=None):
        full_feats_dict = torch.load(protein_feat_file)
        self.data = []

        df = pd.read_csv(csv_path, low_memory=False)

        for _, row in df.iterrows():
            subunit_1 = str(row["subunit_1"]).strip()
            subunit_2 = str(row["subunit_2"]).strip()
            uid = subunit_1
            rna_seq = str(row["s2_sequence"]).strip().upper()
            prot_seq = str(row["s1_sequence"]).strip().upper()

            if allowed_ids is not None and uid not in allowed_ids:
                continue
            if rna_seq == "NAN":
                continue
            if not (config.min_len <= len(rna_seq) <= config.max_len - 2):
                continue
            if uid not in full_feats_dict:
                continue

            vocab = config.rna_vocab
            try:
                tok_body = [vocab[c] for c in rna_seq]
            except KeyError:
                continue

            tgt = torch.tensor(
                [vocab["<sos>"]] + tok_body + [vocab["<eos>"]],
                dtype=torch.long,
            )
            protein_feat = torch.as_tensor(full_feats_dict[uid]).float()

            complex_id = f"{subunit_1}__{subunit_2}"
            self.data.append((protein_feat, tgt, prot_seq, complex_id, rna_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        protein_feat, tgt, prot_seq, complex_id, rna_seq = self.data[idx]
        return protein_feat, tgt, prot_seq, complex_id, rna_seq


def custom_collate_fn_AR(batch):
    pad_id = config.rna_vocab["<pad>"]

    protein_feats, tgt_seqs, prot_seqs, complex_ids, rna_seqs = zip(*batch)

    B = len(protein_feats)
    D = protein_feats[0].size(1)
    S_max = max(feat.size(0) for feat in protein_feats)

    protein_batch = torch.zeros(B, S_max, D, dtype=torch.float32)
    for i, feat in enumerate(protein_feats):
        S_i = feat.size(0)
        protein_batch[i, :S_i] = feat

    # --- RNA ターゲット: PAD で右詰めパディング ---
    maxL = max(t.size(0) for t in tgt_seqs)
    tgt_padded = torch.full((B, maxL), pad_id, dtype=torch.long)
    for i, t in enumerate(tgt_seqs):
        L = t.size(0)
        tgt_padded[i, :L] = t

    return protein_batch, tgt_padded, list(prot_seqs), list(complex_ids), list(rna_seqs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", dest="ckpt", required=True)
    parser.add_argument("--protein_feat_path", dest="protein_feat_path", required=True)
    parser.add_argument("--csv_path", dest="csv_path", required=True)
    parser.add_argument("--output_path", dest="output_path", required=True)
    args = parser.parse_args()

    CKPT_PATH = args.ckpt
    protein_feat_path = args.protein_feat_path
    csv_path = args.csv_path
    output_path = args.output_path

    # --- データ準備 ---

    df = pd.read_csv(csv_path, low_memory=False)

    # ppi3dから取得したデータを使用し、カラム名もそのまま使用
    df["cluster_id"] = df["s1_binding_site_cluster_data_40_area"].apply(lambda x: str(x).split("_")[0])

    # クラスタごとに構造IDをまとめる
    cluster_dict = defaultdict(list)
    for _, row in df.iterrows():
        cluster_dict[row["cluster_id"]].append(row["subunit_1"])

    #clusters = parse_clstr(config.clstr_path)
    clusters = list(cluster_dict.values())
    random.seed(42)
    random.shuffle(clusters)
    split_idx = int(0.95 * len(clusters))
    train_ids = {sid for cluster in clusters[:split_idx] for sid in cluster}
    test_ids = {sid for cluster in clusters[split_idx:] for sid in cluster}

    dataset_train = RNADataset_AR(protein_feat_path, csv_path, allowed_ids=train_ids)
    dataset_test = RNADataset_AR(protein_feat_path, csv_path, allowed_ids=test_ids)

    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn_AR)

    print(f"Trainデータ数: {len(dataset_train)}")
    print(f"Testデータ数: {len(dataset_test)}")

    # --- モデル定義 ---
    model = ProteinToRNA(input_dim=config.input_dim, num_layers=config.num_layers)
    model = nn.DataParallel(model)
    model = model.to(config.device)

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
    stdout_writer = csv.writer(sys.stdout)
    stdout_writer.writerow([
        "complex_id",
        "protein_sequence",
        "generated_rna_sequence",
        "true_rna_sequence",
    ])
    sys.stdout.flush()
    with torch.inference_mode():
        for batch in tqdm(test_loader):
            # custom_collate_fn_AR が (protein_batch, tgt_padded, prot_seqs, complex_ids, true_rna_seqs) を返す想定
            protein_batch, _, prot_seqs, complex_ids, true_rna_seqs = batch

            out_all = sample_decode_multi_AR(
                model,
                protein_batch,                 # [B, S, D]
                max_len=config.max_len,
                num_samples=num_samples,       # 1タンパク質あたり100本
                top_k=config.top_k,
                temperature=config.temp,
            )
            for i, cid in enumerate(complex_ids):
                protein_seq = str(prot_seqs[i]).strip()
                true_rna_seq = str(true_rna_seqs[i]).strip()

                seq_block = out_all[i]
                # num_samples==1 のときは「1本の配列(List[int])」が返るので揃える
                if len(seq_block) > 0 and isinstance(seq_block[0], int):
                    seq_list = [seq_block]
                else:
                    seq_list = seq_block

                for seq_ids in seq_list:
                    if not seq_ids:
                        continue
                    rna_seq = "".join(id2tok[t] for t in seq_ids)
                    rows.append({
                        "complex_id": cid,
                        "protein_sequence": protein_seq,
                        "generated_rna_sequence": rna_seq,
                        "true_rna_sequence": true_rna_seq,
                    })
                    stdout_writer.writerow([
                        cid,
                        protein_seq,
                        rna_seq,
                        true_rna_seq,
                    ])
                sys.stdout.flush()

    out_df = pd.DataFrame(
        rows,
        columns=[
            "complex_id",
            "protein_sequence",
            "generated_rna_sequence",
            "true_rna_sequence",
        ],
    )
    out_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ")
