import random
import re
import subprocess
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Optional, Tuple

import config
from dataset import RNADataset_AR, custom_collate_fn_AR
from decode import sample_decode_multi_AR
from model import ProteinToRNA
from predict import _ids_to_string


def _gc_content(seq: str) -> Optional[float]:
    if not seq:
        return None
    gc = sum(1 for c in seq.upper() if c in ("G", "C"))
    return gc / len(seq)


def _parse_first_number(text: str) -> Optional[float]:
    match = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _rnafold_energy(seq: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (MFE, ensemble_free_energy) for a sequence using RNAfold.
    If RNAfold is unavailable or fails, returns (None, None).
    """
    if not seq:
        return None, None

    try:
        proc = subprocess.run(
            ["RNAfold", "--noPS"],
            input=seq + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        print("RNAfold が見つかりませんでした。MFE/ensembleは空欄になります。", flush=True)
        return None, None
    except subprocess.CalledProcessError as exc:
        print(f"RNAfold 実行に失敗しました: {exc}", flush=True)
        return None, None

    mfe = None
    ensemble = None
    for line in proc.stdout.splitlines():
        if "(" in line and ")" in line:
            val = _parse_first_number(line)
            if val is not None:
                mfe = val
        if "free energy of ensemble" in line:
            val = _parse_first_number(line)
            if val is not None:
                ensemble = val
    return mfe, ensemble

if __name__ == "__main__":
    # --- データ準備（クラスタでtrain/test分割のうちtestのみ使用） ---
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
    test_ids = {sid for cluster in clusters[split_idx:] for sid in cluster}

    dataset_test = RNADataset_AR(config.protein_feat_path, config.csv_path, allowed_ids=test_ids)

    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn_AR)

    print(f"Testデータ数: {len(dataset_test)}")

    # --- モデル定義 & 学習済み重み読込 ---
    base_dir = Path(__file__).resolve().parent
    weights_dir = base_dir / "weights"
    weight_path = weights_dir / "t30_150M_decoder_AR_1129.pt"  # 利用したい重みに合わせて変更

    model = ProteinToRNA(input_dim=config.input_dim, num_layers=config.num_layers)
    model = nn.DataParallel(model).to(config.device)
    state = torch.load(weight_path, map_location=config.device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # --- 生成 ---
    out_dir = base_dir / "generated_rna"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "DecoderOnly_result.csv"

    results = []
    with torch.no_grad():
        for protein_batch, tgt_batch, prot_seq_batch in test_loader:
            protein_batch = protein_batch.to(config.device)

            pred_ids_batch = sample_decode_multi_AR(
                model,
                protein_batch,
                num_samples=config.num_samples,
                max_len=config.max_len,
                top_k=config.top_k,
                temperature=config.temp,
            )

            for prot_seq, tgt_seq, pred_ids in zip(prot_seq_batch, tgt_batch, pred_ids_batch):
                true_seq = _ids_to_string(tgt_seq)
                pred_seq = _ids_to_string(pred_ids)
                pred_len = len(pred_seq)
                pred_gc = _gc_content(pred_seq)
                pred_mfe, pred_ensemble = _rnafold_energy(pred_seq)
                results.append(
                    {
                        "protein_seq": prot_seq,
                        "true_rna_seq": true_seq,
                        "pred_rna_seq": pred_seq,
                        "gc": pred_gc,
                        "length": pred_len,
                        "MFE": pred_mfe,
                        "EFE": pred_ensemble,
                    }
                )

    pd.DataFrame(
        results,
        columns=[
            "protein_seq",
            "true_rna_seq",
            "pred_rna_seq",
            "gc",
            "length",
            "MFE",
            "EFE",
        ],
    ).to_csv(out_path, index=False)
    print(f"予測結果を {out_path} に保存しました。")
