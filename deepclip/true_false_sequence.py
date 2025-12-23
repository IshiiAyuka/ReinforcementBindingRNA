#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import sys
import os
import csv
import numpy as np
import network


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
    sys.stderr.write(str(msg) + "\n")
    sys.stderr.flush()


def oprint(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def open_csv_read(path):
    return open(path, "r")


def detect_delimiter(path):
    try:
        with open_csv_read(path) as fh:
            head = fh.readline()
    except Exception:
        return ","
    if "\t" in head and head.count("\t") >= head.count(","):
        return "\t"
    return ","


def find_model_path(weights_dir, pid, suffix):
    direct = os.path.join(weights_dir, pid + suffix)
    if os.path.isfile(direct):
        return direct
    alt = os.path.join(weights_dir, pid + "_RNCMPT" + suffix)
    if os.path.isfile(alt):
        return alt
    # fallback: any file starting with id
    try:
        for fn in os.listdir(weights_dir):
            if fn.startswith(pid) and fn.endswith(suffix):
                return os.path.join(weights_dir, fn)
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="id×bucket(top/bottom)ごとにDeepCLIPスコア>=thrの割合を算出"
    )
    parser.add_argument("-u", "--protein_csv", dest="protein_csv_opt", help="互換用(未使用) 旧: タンパク質CSV")
    parser.add_argument("-g", "--rna_csv", dest="rna_csv_opt", help="互換用: 入力CSV（id, protein_name, bucket, order, sequence, metric_value）")
    parser.add_argument("-i", "--input_csv", dest="input_csv_opt", help="入力CSV（id, protein_name, bucket, order, sequence, metric_value）")
    parser.add_argument("-w", "--weights", dest="weights_opt", help="DeepCLIP .pkl ディレクトリ")
    parser.add_argument("input_csv", nargs="?", help="入力CSV（id, protein_name, bucket, order, sequence, metric_value）")
    parser.add_argument("weights", nargs="?", help="DeepCLIP .pkl ディレクトリ")
    parser.add_argument("--thr", type=float, default=0.75, help="合格(>=thr)の閾値（既定: 0.75）")
    parser.add_argument("--pkl_suffix", default=".pkl", help="モデルファイルの拡張子（既定: .pkl）")
    args = parser.parse_args()

    input_csv = args.input_csv_opt or args.rna_csv_opt or args.input_csv
    weights_dir = args.weights_opt or args.weights

    if not (input_csv and weights_dir):
        eprint("usage: <script> [-i input.csv] [-w weights_dir] [--thr 0.75]")
        sys.exit(2)

    delimiter = detect_delimiter(input_csv)

    # id -> bucket -> {"protein_name": str, "seqs": [seq,...]}
    try:
        with open_csv_read(input_csv) as fg:
            reader = csv.DictReader(fg, delimiter=delimiter)
            data = {}
            for row in reader:
                pid = (row.get("id") or "").strip()
                name = (row.get("protein_name") or "").strip()
                bucket = (row.get("bucket") or "").strip()
                seq = (row.get("sequence") or "").strip().upper()
                if not (pid and bucket and seq):
                    continue
                entry = data.setdefault(pid, {}).setdefault(bucket, {"protein_name": name, "seqs": []})
                if name and not entry["protein_name"]:
                    entry["protein_name"] = name
                entry["seqs"].append(seq)
    except Exception as e:
        eprint("[ERROR] 入力CSVの読み込みに失敗: {0}".format(e))
        sys.exit(1)

    for pid, bucket_map in sorted(data.items()):
        model_path = find_model_path(weights_dir, pid, args.pkl_suffix)
        if not model_path:
            eprint("[WARN] モデルファイルが見つかりません: id={0}".format(pid))
            continue

        try:
            net, _ = network.load_network(model_path)
            predict_fn, outpar = net.compile_prediction_function()
            options = net.options
            output_shape = net.network["l_in"].output_shape
        except Exception as e:
            eprint("[ERROR] DeepCLIPモデルのロードに失敗: {0} ({1})".format(model_path, e))
            continue

        for bucket, info in sorted(bucket_map.items()):
            seq_list = info["seqs"]
            if not seq_list:
                continue
            name = info["protein_name"]

            n_pass = 0
            n_total = 0
            for seq in seq_list:
                n_total += 1
                try:
                    inputs = seq_to_onehot(seq, options)
                    result = network.predict_without_network(
                        predict_fn, options, output_shape, inputs, outpar
                    )
                    score = float(result["predictions"][0])
                    if score >= args.thr:
                        n_pass += 1
                except Exception as e:
                    eprint("[ERROR] 評価失敗 pid={0} bucket={1} name={2}: {3}".format(pid, bucket, name, e))

            ratio = n_pass / float(n_total) if n_total else 0.0
            # protein_nameをスコアに併記（カンマ区切り）
            oprint("{0}\t{1}\t{2},{3:.3f}".format(pid, bucket, name, ratio))


if __name__ == "__main__":
    main()
