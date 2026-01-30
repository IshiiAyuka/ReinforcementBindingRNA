#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import os
import random
import sys

import numpy as np

DEFAULT_INPUT = "/home/slab/ishiiayuka/M2/deepclip/LucaOneOnly_RNAcompete.csv"
DEFAULT_OUTPUT = "-"
DEFAULT_WEIGHTS_DIR = "/home/slab/ishiiayuka/M2/deepclip/models/RNCMPT"
DEFAULT_OFFTARGET_PROTEIN_CSV = "/home/slab/ishiiayuka/M2/deepclip/RNAcompete.csv"
DEFAULT_ID_COL = "id"
DEFAULT_SEQ_COL = "sequence"
DEFAULT_SCORE_COL = "DeepCLIP_Score"
DEFAULT_OFFTARGET_MEAN_COL = "DeepCLIP_Offtarget_Mean"
DEFAULT_OFFTARGET_N_COL = "DeepCLIP_Offtarget_N"
DEFAULT_SAMPLE_SIZE = 54
DEFAULT_SEED = 42


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


def list_model_ids(weights_dir):
    ids = []
    try:
        for name in os.listdir(weights_dir):
            if name.endswith(".pkl"):
                ids.append(name[:-4])
    except Exception:
        return []
    return sorted(set(ids))


def load_offtarget_ids_from_csv(csv_path):
    try:
        fin = open(csv_path, "r")
    except Exception:
        return []
    try:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            fin.close()
            return []
        if "file_name" not in reader.fieldnames:
            fin.close()
            return []
        ids = []
        for row in reader:
            name = (row.get("file_name") or "").strip()
            if not name:
                continue
            if name.endswith(".pkl"):
                name = name[:-4]
            ids.append(name)
        fin.close()
        return sorted(set(ids))
    except Exception:
        fin.close()
        return []


def stable_seed(base_seed, protein_id, row_index):
    key = "{0}:{1}:{2}".format(base_seed, protein_id, row_index).encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    return int(digest[:8], 16)


def main():
    parser = argparse.ArgumentParser(
        description="DeepCLIPスコアとオフターゲット平均を算出してCSVに追加します"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="入力CSV")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="出力CSV（- で標準出力）"
    )
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
    parser.add_argument(
        "--offtarget-protein-csv",
        default=DEFAULT_OFFTARGET_PROTEIN_CSV,
        help="オフターゲット候補のタンパク質CSV（file_name列を使用）",
    )
    parser.add_argument(
        "--offtarget-mean-col",
        default=DEFAULT_OFFTARGET_MEAN_COL,
        help="追加するオフターゲット平均列名",
    )
    parser.add_argument(
        "--offtarget-n-col",
        default=DEFAULT_OFFTARGET_N_COL,
        help="追加するオフターゲット件数列名",
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="オフターゲット数"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="乱数シード(再現性用)"
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
    for col in (args.score_col, args.offtarget_mean_col, args.offtarget_n_col):
        if col not in out_fieldnames:
            out_fieldnames.append(col)

    if args.output == "-":
        fout = sys.stdout
    else:
        try:
            fout = open(args.output, "w")
        except Exception as e:
            fin.close()
            raise SystemExit("出力CSVの作成に失敗: {0}".format(e))

    writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
    writer.writeheader()

    available_ids = load_offtarget_ids_from_csv(args.offtarget_protein_csv)
    if not available_ids:
        available_ids = list_model_ids(args.weights_dir)
    if not available_ids:
        fin.close()
        fout.close()
        raise SystemExit(
            "オフターゲット候補が見つかりません: {0}".format(
                args.offtarget_protein_csv
            )
        )

    cache = {}
    missing = 0
    load_errors = 0
    predict_errors = 0
    off_target_empty = 0

    for row_index, row in enumerate(reader):
        protein_id = str(row.get(args.id_col, "")).strip()
        seq = str(row.get(args.seq_col, "")).strip().upper()

        # ターゲットスコア
        model_path = resolve_model_path(args.weights_dir, protein_id)
        if not model_path:
            row[args.score_col] = ""
            missing += 1
        else:
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
                    sys.stderr.write(
                        "[WARN] モデル読み込み失敗: {0} ({1})\n".format(model_path, e)
                    )
                else:
                    try:
                        predict_fn, outpar, options, output_shape = cache[model_path]
                        inputs = seq_to_onehot(seq, options)
                        result = network.predict_without_network(
                            predict_fn, options, output_shape, inputs, outpar
                        )
                        row[args.score_col] = "{0:.6f}".format(
                            float(result["predictions"][0])
                        )
                    except Exception as e:
                        predict_errors += 1
                        row[args.score_col] = ""
                        sys.stderr.write(
                            "[WARN] スコア計算失敗 id={0}: {1}\n".format(protein_id, e)
                        )
            else:
                cached = cache.get(model_path)
                if cached is None:
                    row[args.score_col] = ""
                else:
                    try:
                        predict_fn, outpar, options, output_shape = cached
                        inputs = seq_to_onehot(seq, options)
                        result = network.predict_without_network(
                            predict_fn, options, output_shape, inputs, outpar
                        )
                        row[args.score_col] = "{0:.6f}".format(
                            float(result["predictions"][0])
                        )
                    except Exception as e:
                        predict_errors += 1
                        row[args.score_col] = ""
                        sys.stderr.write(
                            "[WARN] スコア計算失敗 id={0}: {1}\n".format(protein_id, e)
                        )

        # オフターゲット平均
        candidates = [mid for mid in available_ids if mid != protein_id]
        if not candidates:
            row[args.offtarget_mean_col] = ""
            row[args.offtarget_n_col] = "0"
            off_target_empty += 1
            writer.writerow(row)
            continue

        rng = random.Random(stable_seed(args.seed, protein_id, row_index))
        if len(candidates) <= args.sample_size:
            sample_ids = candidates
        else:
            sample_ids = rng.sample(candidates, args.sample_size)

        scores = []
        for mid in sample_ids:
            mp = resolve_model_path(args.weights_dir, mid)
            if not mp:
                continue
            if mp not in cache:
                try:
                    net, _ = network.load_network(mp)
                    predict_fn, outpar = net.compile_prediction_function()
                    options = net.options
                    output_shape = net.network["l_in"].output_shape
                    cache[mp] = (predict_fn, outpar, options, output_shape)
                except Exception as e:
                    cache[mp] = None
                    load_errors += 1
                    sys.stderr.write(
                        "[WARN] モデル読み込み失敗: {0} ({1})\n".format(mp, e)
                    )
                    continue
            cached = cache.get(mp)
            if cached is None:
                continue
            try:
                predict_fn, outpar, options, output_shape = cached
                inputs = seq_to_onehot(seq, options)
                result = network.predict_without_network(
                    predict_fn, options, output_shape, inputs, outpar
                )
                scores.append(float(result["predictions"][0]))
            except Exception as e:
                predict_errors += 1
                sys.stderr.write(
                    "[WARN] スコア計算失敗 id={0}: {1}\n".format(mid, e)
                )

        if scores:
            row[args.offtarget_mean_col] = "{0:.6f}".format(sum(scores) / len(scores))
            row[args.offtarget_n_col] = str(len(scores))
        else:
            row[args.offtarget_mean_col] = ""
            row[args.offtarget_n_col] = "0"
            off_target_empty += 1

        writer.writerow(row)

    fin.close()
    if fout is not sys.stdout:
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
    if off_target_empty:
        sys.stderr.write(
            "[WARN] オフターゲットが空の行: {0}\n".format(off_target_empty)
        )
    sys.stderr.write(
        "[DONE] output={0} rows_processed={1}\n".format(
            args.output, row_index + 1 if "row_index" in locals() else 0
        )
    )
    sys.stderr.flush()


if __name__ == "__main__":
    main()
