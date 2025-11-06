#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import csv
import numpy as np
import network  # network.py が同じディレクトリ、または PYTHONPATH 上にあること

def seq_to_onehot(seq, options):
    """
    seq: 文字列シーケンス
    options: net.options から取得
    戻り値 shape=(1,1,VOCAB_SIZE*SEQ_SIZE) の numpy.float32
    """
    # １）モデルの VOCAB を大文字にもマッピング
    vocab = options["VOCAB"]         # 例: ['a','c','g','u','n']
    vocab_upper = [c.upper() for c in vocab]
    # ２）辞書を大文字・小文字の両方で作る
    mapping = {}
    for i, c in enumerate(vocab):
        mapping[c] = i
        mapping[c.upper()] = i

    L   = int(options["SEQ_SIZE"])
    V   = len(vocab)
    arr = np.zeros((1, 1, V * L), dtype=np.float32)

    # ３）実際に one-hot 埋め
    for i, base in enumerate(seq[:L]):
        j = mapping.get(base)
        if j is not None:
            arr[0, 0, i*V + j] = 1.0
        # mapping にない文字 (ギャップや N 以外) は全部ゼロでスキップ

    return arr
def evaluate_model(pkl_path, seq):
    """
    1ファイル・1配列の評価を行いスコアを返す
    """
    # 1) 重みとオプション読み込み
    net, freq = network.load_network(pkl_path)
    # 2) 予測関数コンパイル
    predict_fn, outpar = net.compile_prediction_function()
    options      = net.options
    output_shape = net.network['l_in'].output_shape
    # 3) one-hot 化
    inputs = seq_to_onehot(seq, options)
    # 4) スコア計算
    result = network.predict_without_network(
        predict_fn, options, output_shape, inputs, outpar
    )
    return float(result["predictions"][0])

def main():
    parser = argparse.ArgumentParser(
        description="DeepCLIP 一括評価: uniprot_sequences.csv と generated_rna.csv を参照"
    )
    parser.add_argument(
        "-u", "--uniprot-csv", default="RNCMPT_sequences.csv",
        help="protein モデル情報を含む CSV (id,protein_name,file_name)"
    )
    parser.add_argument(
        "-g", "--generated-csv", default="generated_rna.csv",
        help="RNA配列を含む CSV (id,sequence)"
    )
    parser.add_argument(
        "-w", "--weights-dir", default="models/RNCMPT",
        help="モデル pkl を格納したディレクトリ"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="出力 CSV (省略時は標準出力)"
    )
    args = parser.parse_args()

    # generated_rna.csv を読み込んで id->sequence マップを作成
    try:
        with open(args.generated_csv, "r") as fg:
            gen_reader = csv.DictReader(fg)
            rna_map = { row["id"].strip(): row["sequence"].strip()
                        for row in gen_reader
                        if row.get("id") and row.get("sequence") }
    except Exception as e:
        sys.stderr.write("csvの読み込みに失敗: %s\n" % e)
        sys.exit(1)

    # uniprot_sequences.csv の読み込み
    try:
        fin = open(args.uniprot_csv, "r")
    except Exception as e:
        sys.stderr.write("csvの読み込みに失敗: %s\n" % e)
        sys.exit(1)
    uni_reader = csv.DictReader(fin)

    # 出力先準備
    if args.output:
        fout = open(args.output, "w")
    else:
        fout = sys.stdout
    writer = csv.writer(fout)
    writer.writerow(["uniptrot_id", "protein_name", "score"])

    # 各行ループ
    for row in uni_reader:
        fn   = row.get("file_name", "").strip()
        name = row.get("protein_name", "").strip()
        pid  = row.get("uniprot_id", "").strip()

        if not (pid and name and fn):
            print("情報が不足しているためスキップ:", row)
            continue

        seq = rna_map.get(pid)
        if seq is None:
            sys.stderr.write("RNA配列が見つかりません (id=%s)\n" % pid)
            continue

        pkl_path = os.path.join(args.weights_dir, fn)
        if not os.path.isfile(pkl_path):
            sys.stderr.write("モデルファイルが見つかりません: %s\n" % pkl_path)
            continue

        try:
            score = evaluate_model(pkl_path, seq)
        except Exception as e:
            sys.stderr.write("評価失敗 [id=%s name=%s]: %s\n" % (pid, name, e))
            continue

        writer.writerow([pid, name, "%.6f" % score])
        if fout is sys.stdout:
            sys.stdout.flush()

    fin.close()
    if args.output:
        fout.close()

if __name__ == "__main__":
    main()
