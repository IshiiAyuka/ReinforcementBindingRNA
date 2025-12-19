import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from typing import Optional, Tuple

import config
from dataset import RNADataset_AR, custom_collate_fn_AR
from decode import sample_decode_multi_AR
from model import ProteinToRNA
from predict import _ids_to_string


_RNAFOLD_BIN_CACHE: Optional[str] = None


def _resolve_rnafold_bin() -> Optional[str]:
    """
    RNAfoldの実体を探す。
    優先順位:
      1. config.rnafold_bin
      2. 環境変数 RNAFOLD_BIN
      3. 現在のPATHでの command -v
      4. login shell 経由の command -v (bash -lc)
    見つからなければ None を返す。
    """
    global _RNAFOLD_BIN_CACHE
    if _RNAFOLD_BIN_CACHE:
        return _RNAFOLD_BIN_CACHE

    candidates = [
        getattr(config, "rnafold_bin", None),
        os.environ.get("RNAFOLD_BIN"),
        "RNAfold",
    ]

    for cand in candidates:
        if not cand:
            continue
        if shutil.which(cand):
            _RNAFOLD_BIN_CACHE = cand
            return cand

    # login shell を経由して command -v RNAfold を試す（.bashrc の module load 等が効く場合に検出）
    try:
        res = subprocess.run(
            ["bash", "-lc", "command -v RNAfold"],
            capture_output=True,
            text=True,
            check=True,
        )
        path = res.stdout.strip()
        if path:
            _RNAFOLD_BIN_CACHE = path
            return path
    except Exception:
        pass

    if not hasattr(_resolve_rnafold_bin, "_warned"):
        print("RNAfold が見つかりませんでした。PATH もしくは config.rnafold_bin / 環境変数 RNAFOLD_BIN を設定してください。", flush=True)
        _resolve_rnafold_bin._warned = True

    return None


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

    rnafold_bin = _resolve_rnafold_bin()
    if rnafold_bin is None:
        return None, None

    try:
        # -p でパーティション関数計算を有効化し、アンサンブル自由エネルギーを出力させる
        proc = subprocess.run(
            [rnafold_bin, "-p", "--noPS"],
            input=seq + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        if not hasattr(_rnafold_energy, "_warned"):
            print("RNAfold が見つかりませんでした。MFE/ensembleは空欄になります。", flush=True)
            _rnafold_energy._warned = True
        return None, None
    except subprocess.CalledProcessError as exc:
        print(f"RNAfold 実行に失敗しました: {exc}", flush=True)
        return None, None

    # stdoutとstderr両方を見る（環境により出力先が異なるため）
    out_text = "\n".join(filter(None, [proc.stdout, proc.stderr]))

    mfe = None
    ensemble = None
    energies = []
    for line in out_text.splitlines():
        # 任意の括弧 () [] {} に挟まれた最初の数値を拾う
        m_energy = re.search(r"[\(\[\{]\s*([-+]?\d+(?:\.\d+)?)", line)
        if m_energy:
            val = float(m_energy.group(1))
            energies.append(val)
            if "(" in line and mfe is None:
                mfe = val

        # 明示的な ensemble 行があればそれを採用
        m_ens = re.search(r"ensemble[^=]*=\s*([-+]?\d+(?:\.\d+)?)", line, flags=re.IGNORECASE)
        if m_ens:
            ensemble = float(m_ens.group(1))

    # ensemble が取れない場合、最後に出たエネルギー値をフォールバックとして使う
    if ensemble is None and energies:
        ensemble = energies[-1]

    # ensemble が取れない場合、デバッグ用に1回だけ出力
    if ensemble is None and not hasattr(_rnafold_energy, "_warned_no_ensemble"):
        print("RNAfold 出力からアンサンブル自由エネルギーを取得できませんでした。出力を確認してください。", flush=True)
        _rnafold_energy._warned_no_ensemble = True

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
    weight_path = weights_dir / "t30_150M_decoder_AR_reinforce_LucaOneOnly.pt"  # 利用したい重みに合わせて変更

    state = torch.load(weight_path, map_location=config.device)
    base_model = ProteinToRNA(input_dim=config.input_dim, num_layers=config.num_layers)
    base_model.load_state_dict(state, strict=True)
    model = nn.DataParallel(base_model).to(config.device)
    model.eval()

    # --- 生成 ---
    out_dir = base_dir / "generated_rna"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "LucaOneOnly_result.csv"

    results = []
    with torch.no_grad():
        for protein_batch, tgt_batch, prot_seq_batch in tqdm(
            test_loader, desc="Generating RNA", total=len(test_loader)
        ):
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
                print(
                    f"[RNA] GC={pred_gc} len={pred_len} MFE={pred_mfe} EFE={pred_ensemble} seq={pred_seq}",
                    flush=True,
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
