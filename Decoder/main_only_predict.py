import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from dataset import RNADataset, custom_collate_fn
from decode import sample_decode
from model import ProteinToRNA
import config
from collections import defaultdict
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # --- データ準備 ---
    df = pd.read_csv(config.csv_path, low_memory=False)

    pdb_id_dict   = dict(zip(df["subunit_1"], df["pdb_id"]))
    prot_dict     = dict(zip(df["subunit_1"], df["s1_sequence"]))
    true_rna_dict = dict(zip(df["subunit_1"], df["s2_sequence"]))
    df["cluster_id"] = df["s1_binding_site_cluster_data_40_area"].apply(lambda x: str(x).split("_")[0])
    cluster_dict = defaultdict(list)
    for _, row in df.iterrows():
        cluster_dict[row["cluster_id"]].append(row["subunit_1"])
    #clusters = parse_clstr(config.clstr_path)
    clusters = list(cluster_dict.values())
    random.seed(42)
    random.shuffle(clusters)
    split_idx = int(0.95 * len(clusters))
    test_ids = {sid for cluster in clusters[split_idx:] for sid in cluster}

    dataset_test = RNADataset(config.protein_feat_path, config.csv_path, allowed_ids=test_ids)
    #test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    print(f"Testデータ数: {len(dataset_test)}")

    # --- モデル定義 ---
    model = ProteinToRNA(input_dim=config.input_dim, num_layers=config.num_layers)
    model = nn.DataParallel(model).to(config.device)
    model.load_state_dict(torch.load("t6_8M_decoder_trimmed.pt", map_location=config.device))
    model.eval()

    results = []
    with torch.no_grad():
        for feat, tgt, uid in tqdm(dataset_test, desc="評価中"):
            pdb_id   = pdb_id_dict.get(uid, uid)
            prot_seq = prot_dict[uid]
            true_seq = true_rna_dict[uid]

            feat = feat.to(config.device)

            pred_ids = sample_decode(model, feat, max_len=config.max_len, num_samples=config.num_samples)[0]
            pred_seq = "".join(config.rna_ivocab[i] for i in pred_ids)

            results.append({"pdb_id": pdb_id, "protein_seq": prot_seq, "true_rna_seq": true_seq, "pred_rna_seq": pred_seq})


    df_out = pd.DataFrame(results,columns=["pdb_id", "protein_seq", "true_rna_seq", "pred_rna_seq"])
    out_path = "predictions_trimmed.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"{out_path} に結果を保存しました。")


