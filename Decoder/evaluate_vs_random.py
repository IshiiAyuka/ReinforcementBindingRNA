#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import RNADataset, custom_collate_fn
from utils import global_alignment, local_alignment

# A/U/C/G のIDだけ使用
BASE_TOKEN_IDS = [config.rna_vocab["A"], config.rna_vocab["U"],
                  config.rna_vocab["C"], config.rna_vocab["G"]]

def sample_random_ids(target_len=500):
    """1〜target_len のランダム長の A/U/C/G 列を返す"""
    target_len = max(1, int(target_len))
    L = random.randint(1, target_len)
    return [random.choice(BASE_TOKEN_IDS) for _ in range(L)]

def main():
    # データ読み込み（全件）
    ds = RNADataset(config.protein_feat_path, config.csv_path, allowed_ids=None)
    loader = DataLoader(
        ds,
        batch_size=getattr(config, "batch_size", 32),
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )

    pad_id = config.rna_vocab["<pad>"]
    sos_id = config.rna_vocab["<sos>"]
    eos_id = config.rna_vocab["<eos>"]

    recalls, globals_, locals_ = [], [], []

    with torch.no_grad():
        for protein_batch, tgt_batch, _uids in tqdm(loader, desc="ランダム配列で評価中"):
            B = tgt_batch.size(0)
            for i in range(B):
                # --- ターゲットから <sos>/<eos>/<pad> を除去 ---
                seq = tgt_batch[i].tolist()
                if seq and seq[0] == sos_id:
                    seq = seq[1:]
                try:
                    seq = seq[:seq.index(eos_id)]
                except ValueError:
                    pass
                target_ids = [x for x in seq if x != pad_id]
                if not target_ids:
                    continue

                # --- ランダム列作成 ---
                rnd_ids = sample_random_ids()

                # --- 文字列へ変換 ---
                rnd_seq    = "".join(config.rna_ivocab[int(t)] for t in rnd_ids)
                target_seq = "".join(config.rna_ivocab[int(t)] for t in target_ids)

                # --- Recall（先頭からの位置一致 / ターゲット長） ---
                m = min(len(rnd_ids), len(target_ids))
                match = sum(rnd_ids[j] == target_ids[j] for j in range(m))
                recalls.append(match / len(target_ids))

                # --- アライメント ---
                globals_.append(global_alignment(rnd_seq, target_seq))
                locals_.append(local_alignment(rnd_seq, target_seq))

    avg_recall = sum(recalls)/len(recalls) if recalls else 0.0
    avg_global = sum(globals_)/len(globals_) if globals_ else 0.0
    avg_local  = sum(locals_)/len(locals_)  if locals_  else 0.0

    print(f"【ランダム比較】平均 Recall: {avg_recall:.4f}", flush=True)
    print(f"【ランダム比較】平均 Global:  {avg_global:.4f}", flush=True)
    print(f"【ランダム比較】平均 Local:   {avg_local:.4f}", flush=True)

if __name__ == "__main__":
    main()
