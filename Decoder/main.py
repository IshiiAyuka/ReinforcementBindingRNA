import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RNADataset_AR, custom_collate_fn_AR
from model import ProteinToRNA
from train import train_model_AR
from evaluate import evaluate_model
from plots import plot_loss
import random
import config
from predict import show_test_samples
from collections import defaultdict
import pandas as pd
import argparse

if __name__ == "__main__":
    # --- データ準備 ---
    parser = argparse.ArgumentParser(description="Train ProteinToRNA model.")
    parser.add_argument("protein_feat_path", help="入力のタンパク質特徴量のptファイルパス")
    parser.add_argument("csv_path", help="入力のCSVファイルパス")
    parser.add_argument("save_model", help="出力のモデル保存パス")
    parser.add_argument("save_lossplot", help="出力のロスプロット保存パス")
    args = parser.parse_args()

    config.protein_feat_path = args.protein_feat_path
    config.csv_path = args.csv_path
    config.save_model = args.save_model
    config.save_lossplot = args.save_lossplot

    df = pd.read_csv(config.csv_path, low_memory=False)

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

    dataset_train = RNADataset_AR(config.protein_feat_path, config.csv_path, allowed_ids=train_ids)
    dataset_test = RNADataset_AR(config.protein_feat_path, config.csv_path, allowed_ids=test_ids)

    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, collate_fn=custom_collate_fn_AR)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn_AR)

    print(f"Trainデータ数: {len(dataset_train)}")
    print(f"Testデータ数: {len(dataset_test)}")

    # --- モデル定義 ---
    model = ProteinToRNA(input_dim=config.input_dim, num_layers=config.num_layers)
    model = nn.DataParallel(model)
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.rna_vocab["<pad>"], reduction="sum")

    # --- 学習 ---
    loss_history = train_model_AR(model, train_loader, optimizer, criterion, config.device, config.epochs)

    # --- モデル保存 ---
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state, config.save_model)
    print("モデルを保存しました。", flush=True)

    # --- ロスプロット ---
    plot_loss(loss_history)

    # --- サンプル表示（Testデータ） ---
    show_test_samples(model, dataset_test, config.device)

    # --- 評価 ---
    print("\n==== Testデータセット評価 ====", flush=True)
    evaluate_model(model, test_loader, config.device) 

    print("\n==== Trainデータセット評価 ====", flush=True)
    evaluate_model(model, train_loader, config.device)
