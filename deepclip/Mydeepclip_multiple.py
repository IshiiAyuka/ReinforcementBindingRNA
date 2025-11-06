#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import csv
import numpy as np
import network  # network.py が同じディレクトリ、または PYTHONPATH 上にあること

def seq_to_onehot(seq, options):

    vocab = options["VOCAB"]
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

def evaluate_model(pkl_path, seq):
    net, freq = network.load_network(pkl_path)
    predict_fn, outpar = net.compile_prediction_function()
    options = net.options
    output_shape = net.network['l_in'].output_shape

    inputs = seq_to_onehot(seq, options)
    result = network.predict_without_network(
        predict_fn, options, output_shape, inputs, outpar
    )
    return float(result["predictions"][0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("protein_csv") #proteinのcsv
    parser.add_argument("rna_csv") #RNAのcsv
    parser.add_argument("weights") #重みのディレクトリ
    parser.add_argument("output", nargs="?")
    args = parser.parse_args()

    try:
        with open(args.rna_csv, "r") as fg:
            gen_reader = csv.DictReader(fg)
            rna_map = {}
            for row in gen_reader:
                pid = row.get("id", "").strip()
                seq = row.get("sequence", "").strip()
                if pid and seq:
                    rna_map.setdefault(pid, []).append(seq)
    except Exception as e:
        sys.stderr.write("csvの読み込みに失敗: %s\n" % e)
        sys.exit(1)

    try:
        fin = open(args.protein_csv, "r")
    except Exception as e:
        sys.stderr.write("csvの読み込みに失敗: %s\n" % e)
        sys.exit(1)
    uni_reader = csv.DictReader(fin)

    if args.output:
        fout = open(args.output, "w")
    else:
        fout = sys.stdout
    writer = csv.writer(fout)
    writer.writerow(["uniprot_id", "protein_name", "score_index", "score"])

    for row in uni_reader:
        fn = row.get("file_name", "").strip()
        name = row.get("protein_name", "").strip()
        pid = row.get("uniprot_id", "").strip()

        if not (pid and name and fn):
            print("情報が不足しているためスキップ: {}".format(row))
            continue

        seq_list = rna_map.get(pid)
        if not seq_list:
            sys.stderr.write("RNA配列が見つかりません (id=%s)\n" % pid)
            continue

        pkl_path = os.path.join(args.weights, fn)
        if not os.path.isfile(pkl_path):
            sys.stderr.write("モデルファイルが見つかりません: %s\n" % pkl_path)
            continue

        scores = []
        for idx, seq in enumerate(seq_list):
            try:
                score = evaluate_model(pkl_path, seq)
                scores.append(score)
                writer.writerow([pid, name, idx + 1, "%.6f" % score])
            except Exception as e:
                sys.stderr.write(
                    "評価失敗 [id=%s name=%s RNA#%d]: %s\n" %
                    (pid, name, idx + 1, e)
                )

        if scores:
            avg_score = sum(scores) / len(scores)
            pass_count = sum(1 for s in scores if s >= 0.75)
            total = len(scores)
            ratio = pass_count / total
            # f文字列を使わず format で出力
            print("[{0}] {1} - スコア平均: {2:.6f}, スコア>=0.75 の割合: {3:.2%} ({4}/{5})".format(
                pid, name, avg_score, ratio, pass_count, total
            ))

    fin.close()
    if args.output:
        fout.close()

if __name__ == "__main__":
    main()
