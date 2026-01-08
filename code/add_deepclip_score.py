#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import csv
import numpy as np

DEFAULT_INPUT = "/home/slab/ishiiayuka/M2/deepclip/All_MFE_RNAcompete.csv"
DEFAULT_OUTPUT = "/home/slab/ishiiayuka/M2/deepclip_result/All_MFE_RNAcompete_with_score.csv"
DEFAULT_WEIGHTS_DIR = "/home/slab/ishiiayuka/M2/deepclip/models/RNCMPT"
DEFAULT_ID_COL = "id"
DEFAULT_SEQ_COL = "sequence"
DEFAULT_SCORE_COL = "DeepCLIP_Score"


def add_deepclip_to_syspath():
    here = os.path.dirname(os.path.abspath(__file__))
    deepclip_dir = os.path.normpath(os.path.join(here, "..", "deepclip"))
    if deepclip_dir not in sys.path:
        sys.path.insert(0, deepclip_dir)


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


def resolve_model_path(weights_dir, protein_id):
    if not protein_id:
        return None
    if protein_id.endswith(".pkl"):
        candidate = os.path.join(weights_dir, protein_id)
        return candidate if os.path.isfile(candidate) else None
    candidate = os.path.join(weights_dir, "{}.pkl".format(protein_id))
    return candidate if os.path.isfile(candidate) else None


def main():
    parser = argparse.ArgumentParser(
        description="DeepCLIPスコアを算出してCSVに追加します"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="入力CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="出力CSV")
    parser.add_argument(
        "--weights-dir",
        default=DEFAULT_WEIGHTS_DIR,
        help="DeepCLIP .pkl ディレクトリ",
    )
    parser.add_argument("--id-col", default=DEFAULT_ID_COL, help="タンパク質ID列名")
    parser.add_argument("--seq-col", default=DEFAULT_SEQ_COL, help="RNA配列列名")
    parser.add_argument(
        "--score-col", default=DEFAULT_SCORE_COL, help="追加するスコア列名"
    )
    args = parser.parse_args()

    add_deepclip_to_syspath()
    try:
        import network
    except Exception as e:
        raise SystemExit("DeepCLIPのnetwork.pyが読み込めません: {0}".format(e))

    try:
        fin = open(args.input, "r")
    except Exception as e:
        raise SystemExit("入力CSVの読み込みに失敗: {0}".format(e))

    reader = csv.DictReader(fin)
    if reader.fieldnames is None:
        fin.close()
        raise SystemExit("入力CSVにヘッダーがありません")

    fieldnames = [name.strip() for name in reader.fieldnames]
    if args.id_col not in fieldnames:
        fin.close()
        raise SystemExit("ID列が見つかりません: {0}".format(args.id_col))
    if args.seq_col not in fieldnames:
        fin.close()
        raise SystemExit("配列列が見つかりません: {0}".format(args.seq_col))

    out_fieldnames = list(fieldnames)
    if args.score_col not in out_fieldnames:
        out_fieldnames.append(args.score_col)

    try:
        fout = open(args.output, "w")
    except Exception as e:
        fin.close()
        raise SystemExit("出力CSVの作成に失敗: {0}".format(e))

    writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
    writer.writeheader()

    cache = {}
    missing = 0
    load_errors = 0
    predict_errors = 0

    for row in reader:
        protein_id = str(row.get(args.id_col, "")).strip()
        seq = str(row.get(args.seq_col, "")).strip().upper()

        model_path = resolve_model_path(args.weights_dir, protein_id)
        if not model_path:
            row[args.score_col] = ""
            writer.writerow(row)
            missing += 1
            continue

        if model_path not in cache:
            try:
                net, _ = network.load_network(model_path)
                predict_fn, outpar = net.compile_prediction_function()
                options = net.options
                output_shape = net.network["l_in"].output_shape
                cache[model_path] = (predict_fn, outpar, options, output_shape)
            except Exception as e:
                cache[model_path] = None
                load_errors += 1
                row[args.score_col] = ""
                writer.writerow(row)
                sys.stderr.write(
                    "[WARN] モデル読み込み失敗: {0} ({1})\n".format(model_path, e)
                )
                continue

        cached = cache.get(model_path)
        if cached is None:
            row[args.score_col] = ""
            writer.writerow(row)
            continue

        try:
            predict_fn, outpar, options, output_shape = cached
            inputs = seq_to_onehot(seq, options)
            result = network.predict_without_network(
                predict_fn, options, output_shape, inputs, outpar
            )
            row[args.score_col] = "{0:.6f}".format(float(result["predictions"][0]))
        except Exception as e:
            predict_errors += 1
            row[args.score_col] = ""
            sys.stderr.write(
                "[WARN] スコア計算失敗 id={0}: {1}\n".format(protein_id, e)
            )
        writer.writerow(row)

    fin.close()
    fout.close()

    if missing:
        sys.stderr.write(
            "[WARN] モデルが見つからずスコア未計算の行: {0}\n".format(missing)
        )
    if load_errors:
        sys.stderr.write(
            "[WARN] モデル読み込み失敗の行: {0}\n".format(load_errors)
        )
    if predict_errors:
        sys.stderr.write(
            "[WARN] スコア計算失敗の行: {0}\n".format(predict_errors)
        )


if __name__ == "__main__":
    main()
