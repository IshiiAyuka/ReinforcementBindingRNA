#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function  
import argparse
import sys
import os
import csv
import numpy as np
import network 

num_samples = 100

def seq_to_onehot(seq, options):

    vocab = options["VOCAB"]
    seq = (seq or "").upper()

    mapping = {c: i for i, c in enumerate(vocab)}
    mapping.update({c.upper(): i for i, c in enumerate(vocab)})

    L = int(options["SEQ_SIZE"])
    V = len(vocab)
    arr = np.zeros((1, 1, V * L), dtype=np.float32)

    for i, base in enumerate(seq[:L]):
        j = mapping.get(base)
        if j is not None:
            arr[0, 0, i * V + j] = 1.0
    return arr

def eprint(msg):
    sys.stderr.write(str(msg) + "\n"); sys.stderr.flush()

def oprint(line):
    sys.stdout.write(line + "\n"); sys.stdout.flush()

def open_csv_read(path):
    return open(path, "r")

def id_from_file_name(fn):
    """例: RNCMPT00001_RNCMPT.pkl -> RNCMPT00001"""
    base = os.path.basename(fn)
    root, _ = os.path.splitext(base)
    #return root.split("_", 1)[0]
    return root

def main():
    parser = argparse.ArgumentParser(
        description="DeepCLIPで生成RNAを逐次評価（IDごとの割合のみ標準出力）"
    )
    # --- 新規: -u/-g/-w でも指定できるように（後方互換） ---
    parser.add_argument("-u", "--protein_csv", dest="protein_csv_opt", help="タンパク質CSV（file_name, protein_name, uniprot_id）")
    parser.add_argument("-g", "--rna_csv",     dest="rna_csv_opt",     help="生成RNA CSV（id, sequence）")
    parser.add_argument("-w", "--weights",     dest="weights_opt",     help="DeepCLIP .pkl ディレクトリ")
    parser.add_argument("protein_csv", nargs="?", help="タンパク質CSV（file_name, protein_name, uniprot_id）")
    parser.add_argument("rna_csv",     nargs="?", help="生成RNA CSV（id, sequence）")
    parser.add_argument("weights",     nargs="?", help="DeepCLIP .pkl ディレクトリ")
    parser.add_argument("--thr", type=float, default=0.75, help="合格(>=thr)の閾値（既定: 0.75）")
    args = parser.parse_args()

    # 実引数を統一（-u/-g/-w 優先、なければ位置引数）
    protein_csv = args.protein_csv_opt or args.protein_csv
    rna_csv     = args.rna_csv_opt     or args.rna_csv
    weights_dir = args.weights_opt     or args.weights

    if not (protein_csv and rna_csv and weights_dir):
        eprint("usage: <script> [-u protein.csv] [-g rna.csv] [-w weights_dir] [--thr 0.75] もしくは位置引数3つ")
        sys.exit(2)

    # id -> [seq,...]
    try:
        with open_csv_read(rna_csv) as fg:
            gen_reader = csv.DictReader(fg)
            rna_map = {}
            for row in gen_reader:
                pid = (row.get("id") or "").strip()
                seq = (row.get("sequence") or "").strip().upper()
                if pid and seq:
                    rna_map.setdefault(pid, []).append(seq)
    except Exception as e:
        eprint("[ERROR] 生成RNA CSVの読み込みに失敗: {0}".format(e))
        sys.exit(1)

    # タンパク質CSVを1行ずつ処理
    try:
        fin = open_csv_read(protein_csv)
    except Exception as e:
        eprint("[ERROR] タンパク質CSVの読み込みに失敗: {0}".format(e))
        sys.exit(1)

    uni_reader = csv.DictReader(fin)

    for row in uni_reader:
        fn   = (row.get("file_name")    or "").strip()
        name = (row.get("protein_name") or "").strip()

        if not fn:
            eprint("[WARN] file_name が空のためスキップ: {0}".format(row))
            continue

        pid = id_from_file_name(fn)

        seq_list = rna_map.get(pid)
        if not seq_list:
            eprint("[WARN] RNA配列が見つかりません (id={0})".format(pid))
            continue

        if len(seq_list) != num_samples:
            eprint("[WARN] id={0} の本数が {1}/{2}".format(pid, len(seq_list), num_samples))

        pkl_path = os.path.join(weights_dir, fn)
        if not os.path.isfile(pkl_path):
            eprint("[WARN] モデルファイルが見つかりません: {0}".format(pkl_path))
            continue

        # DeepCLIPモデルはIDごとに1回だけロード
        try:
            net, _ = network.load_network(pkl_path)
            predict_fn, outpar = net.compile_prediction_function()
            options = net.options
            output_shape = net.network["l_in"].output_shape
        except Exception as e:
            eprint("[ERROR] DeepCLIPモデルのロードに失敗: {0} ({1})".format(pkl_path, e))
            continue

        # 0.75以上だけカウント（失敗は0扱い、分母は常に num_samples）
        n_pass = 0
        max_score = 0.0
        for seq in seq_list[:num_samples]:  # 念のため超過は無視
            try:
                inputs = seq_to_onehot(seq, options)
                result = network.predict_without_network(
                    predict_fn, options, output_shape, inputs, outpar
                )
                score = float(result["predictions"][0])
                if score > max_score:
                    max_score = score
                if score >= args.thr:
                    n_pass += 1
            except Exception as e:
                eprint("[ERROR] 評価失敗 pid={0} name={1}: {2}".format(pid, name, e))

        ratio = n_pass / float(num_samples)  # 常に100で割る
        oprint("{0}\t{1}\t{2:.3f},{3:.3f}".format(pid, name, max_score, ratio))

    fin.close()

if __name__ == "__main__":
    main()
