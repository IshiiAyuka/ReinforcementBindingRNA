import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RNADataset_NAR,RNADataset_AR, custom_collate_fn
from model import ProteinToRNA_NAR, ProteinToRNA
from train import train_model_NAR,train_model_AR
from evaluate import evaluate_model_NAR,evaluate_model
from plots import plot_loss
import random
import config
from predict import show_test_samples_NAR, show_test_samples, sample_decode_multi
from collections import defaultdict
import pandas as pd

if __name__ == "__main__":
    # --- データ準備 ---

    df = pd.read_csv(config.csv_path, low_memory=False)

    # 「s1_binding_site_cluster_data_40_area」列からクラスタ番号を抽出
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

    dataset_train = RNADataset_NAR(config.protein_feat_path, config.csv_path, allowed_ids=train_ids)
    dataset_test = RNADataset_NAR(config.protein_feat_path, config.csv_path, allowed_ids=test_ids)

    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    print(f"Trainデータ数: {len(dataset_train)}")
    print(f"Testデータ数: {len(dataset_test)}")

    # --- モデル定義 ---
    model = ProteinToRNA_NAR(input_dim=config.input_dim, num_layers=config.num_layers)
    model = nn.DataParallel(model)
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.rna_vocab["<pad>"], reduction="sum")

    # --- 学習 ---
    loss_history = train_model_NAR(model, train_loader, optimizer, criterion, config.device, config.epochs)

    # --- モデル保存 ---
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state, config.save_model)
    print("モデルを保存しました。", flush=True)

    # --- ロスプロット ---
    plot_loss(loss_history)

    # --- サンプル表示（Testデータ） ---
    show_test_samples_NAR(model, dataset_test, config.device)

    # --- 評価 ---
    print("\n==== Testデータセット評価 ====", flush=True)
    evaluate_model_NAR(model, test_loader, config.device) 

    print("\n==== Trainデータセット評価 ====", flush=True)
    evaluate_model_NAR(model, train_loader, config.device)

