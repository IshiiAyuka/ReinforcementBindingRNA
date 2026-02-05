#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import os
import random
import sys

import numpy as np
import multiprocessing as mp

DEFAULT_INPUT = "/home/slab/ishiiayuka/M2/deepclip/All_MFE_RNAcompete.csv"
DEFAULT_OUTPUT = "-"
DEFAULT_WEIGHTS_DIR = "/home/slab/ishiiayuka/M2/deepclip/models/RNCMPT"
DEFAULT_OFFTARGET_PROTEIN_CSV = "/home/slab/ishiiayuka/M2/deepclip/RNAcompete.csv"
DEFAULT_ID_COL = "id"
DEFAULT_SEQ_COL = "sequence"
DEFAULT_SCORE_COL = "DeepCLIP_Score_Mean"
DEFAULT_OFFTARGET_MEAN_COL = "DeepCLIP_Offtarget_Mean"
DEFAULT_OFFTARGET_N_COL = "DeepCLIP_Offtarget_N"
DEFAULT_RNA_USED_COL = "RNA_Used_N"
DEFAULT_SAMPLE_SIZE = 54
DEFAULT_OFFTARGET_SAMPLE_SIZE = 10
DEFAULT_MAX_RNA_PER_PROTEIN = 5
DEFAULT_SEED = 42
DEFAULT_WORKERS = 1
DEFAULT_MAXTASKS = 50

G_ARGS = None
G_AVAILABLE_IDS = None
G_CACHE = None
G_NETWORK = None


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
        seen = set()
        for row in reader:
            name = (row.get("file_name") or "").strip()
            if not name:
                continue
            if name.endswith(".pkl"):
                name = name[:-4]
            if name not in seen:
                seen.add(name)
                ids.append(name)
        fin.close()
        return ids
    except Exception:
        fin.close()
        return []


def stable_seed(base_seed, protein_id, row_index):
    key = "{0}:{1}:{2}".format(base_seed, protein_id, row_index).encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    return int(digest[:8], 16)


def init_worker(args_dict, available_ids):
    global G_ARGS, G_AVAILABLE_IDS, G_CACHE, G_NETWORK
    G_ARGS = args_dict
    G_AVAILABLE_IDS = available_ids
    G_CACHE = {}
    add_deepclip_to_syspath()
    try:
        import network  # noqa: F401
    except Exception as e:
        raise SystemExit("DeepCLIPのnetwork.pyが読み込めません: {0}".format(e))
    G_NETWORK = sys.modules.get("network")


def score_row(payload):
    protein_index, protein_id, rows = payload
    args = G_ARGS
    cache = G_CACHE
    network = G_NETWORK

    missing = 0
    load_errors = 0
    predict_errors = 0
    off_target_empty = 0
    used_rows = rows[: args["max_rna_per_protein"]]
    seqs = [str(r.get(args["seq_col"], "")).strip().upper() for r in used_rows]
    seqs = [s for s in seqs if s]

    # ターゲットスコア
    model_path = resolve_model_path(args["weights_dir"], protein_id)
    if not model_path:
        score_values = []
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
                sys.stderr.write(
                    "[WARN] モデル読み込み失敗: {0} ({1})\n".format(model_path, e)
                )
        cached = cache.get(model_path)
        if cached is None:
            score_values = []
        else:
            score_values = []
            for seq in seqs:
                try:
                    predict_fn, outpar, options, output_shape = cached
                    inputs = seq_to_onehot(seq, options)
                    result = network.predict_without_network(
                        predict_fn, options, output_shape, inputs, outpar
                    )
                    score_values.append(float(result["predictions"][0]))
                except Exception as e:
                    predict_errors += 1
                    sys.stderr.write(
                        "[WARN] スコア計算失敗 id={0}: {1}\n".format(protein_id, e)
                    )

    # オフターゲット平均
    candidates = [mid for mid in G_AVAILABLE_IDS if mid != protein_id]
    if args["offtarget_pool_size"] and len(candidates) > args["offtarget_pool_size"]:
        candidates = candidates[: args["offtarget_pool_size"]]
    if not candidates:
        off_target_empty += 1
        out_row = {
            args["id_col"]: protein_id,
            args["score_col"]: (
                "{0:.6f}".format(sum(score_values) / len(score_values))
                if score_values
                else ""
            ),
            args["offtarget_mean_col"]: "",
            args["offtarget_n_col"]: "0",
            args["rna_used_col"]: str(len(seqs)),
        }
        return (
            protein_index,
            out_row,
            missing,
            load_errors,
            predict_errors,
            off_target_empty,
        )

    rng = random.Random(stable_seed(args["seed"], protein_id, protein_index))
    if len(candidates) <= args["offtarget_sample_size"]:
        sample_ids = candidates
    else:
        sample_ids = rng.sample(candidates, args["offtarget_sample_size"])

    scores = []
    models_used = 0
    for mid in sample_ids:
        mpth = resolve_model_path(args["weights_dir"], mid)
        if not mpth:
            continue
        if mpth not in cache:
            try:
                net, _ = network.load_network(mpth)
                predict_fn, outpar = net.compile_prediction_function()
                options = net.options
                output_shape = net.network["l_in"].output_shape
                cache[mpth] = (predict_fn, outpar, options, output_shape)
            except Exception as e:
                cache[mpth] = None
                load_errors += 1
                sys.stderr.write(
                    "[WARN] モデル読み込み失敗: {0} ({1})\n".format(mpth, e)
                )
                continue
        cached = cache.get(mpth)
        if cached is None:
            continue
        models_used += 1
        for seq in seqs:
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

    out_row = {
        args["id_col"]: protein_id,
        args["score_col"]: (
            "{0:.6f}".format(sum(score_values) / len(score_values))
            if score_values
            else ""
        ),
        args["offtarget_mean_col"]: (
            "{0:.6f}".format(sum(scores) / len(scores)) if scores else ""
        ),
        args["offtarget_n_col"]: str(models_used) if scores else "0",
        args["rna_used_col"]: str(len(seqs)),
    }
    if not scores:
        off_target_empty += 1

    return (
        protein_index,
        out_row,
        missing,
        load_errors,
        predict_errors,
        off_target_empty,
    )


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
        "--rna-used-col",
        default=DEFAULT_RNA_USED_COL,
        help="使用RNA本数の列名",
    )
    parser.add_argument(
        "--max-rna-per-protein",
        type=int,
        default=DEFAULT_MAX_RNA_PER_PROTEIN,
        help="1タンパク質あたりに使うRNA本数（先頭から）",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="オフターゲット候補の上限数（自分以外から先頭N件）",
    )
    parser.add_argument(
        "--offtarget-sample-size",
        type=int,
        default=DEFAULT_OFFTARGET_SAMPLE_SIZE,
        help="オフターゲットとしてランダムに選ぶタンパク質数",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="乱数シード(再現性用)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="CPU並列数（1で単一プロセス）",
    )
    parser.add_argument(
        "--maxtasksperchild",
        type=int,
        default=DEFAULT_MAXTASKS,
        help="各ワーカーの最大タスク数（メモリ肥大化対策）",
    )
    args = parser.parse_args()

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

    out_fieldnames = [
        args.id_col,
        args.score_col,
        args.offtarget_mean_col,
        args.offtarget_n_col,
        args.rna_used_col,
    ]

    if args.output == "-":
        fout = sys.stdout
    else:
        try:
            fout = open(args.output, "w")
        except Exception as e:
            fin.close()
            raise SystemExit("出力CSVの作成に失敗: {0}".format(e))

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

    writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
    writer.writeheader()

    rows = list(reader)
    grouped = {}
    order = []
    for row in rows:
        pid = str(row.get(args.id_col, "")).strip()
        if not pid:
            continue
        if pid not in grouped:
            grouped[pid] = []
            order.append(pid)
        if len(grouped[pid]) < args.max_rna_per_protein:
            grouped[pid].append(row)
    tasks = [(i, pid, grouped[pid]) for i, pid in enumerate(order)]
    total_rows = len(tasks)

    args_dict = {
        "weights_dir": args.weights_dir,
        "id_col": args.id_col,
        "seq_col": args.seq_col,
        "score_col": args.score_col,
        "offtarget_mean_col": args.offtarget_mean_col,
        "offtarget_n_col": args.offtarget_n_col,
        "rna_used_col": args.rna_used_col,
        "max_rna_per_protein": args.max_rna_per_protein,
        "offtarget_pool_size": args.sample_size,
        "offtarget_sample_size": args.offtarget_sample_size,
        "seed": args.seed,
    }

    missing = 0
    load_errors = 0
    predict_errors = 0
    off_target_empty = 0

    if args.workers <= 1:
        init_worker(args_dict, available_ids)
        for payload in tasks:
            (
                _idx,
                out_row,
                m,
                lerr,
                perr,
                oempty,
            ) = score_row(payload)
            missing += m
            load_errors += lerr
            predict_errors += perr
            off_target_empty += oempty
            writer.writerow(out_row)
    else:
        pool = mp.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(args_dict, available_ids),
            maxtasksperchild=max(1, int(args.maxtasksperchild)),
        )
        try:
            for (
                _idx,
                out_row,
                m,
                lerr,
                perr,
                oempty,
            ) in pool.imap(score_row, tasks, chunksize=1):
                missing += m
                load_errors += lerr
                predict_errors += perr
                off_target_empty += oempty
                writer.writerow(out_row)
        finally:
            pool.close()
            pool.join()

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
            args.output, total_rows
        )
    )
    sys.stderr.flush()


if __name__ == "__main__":
    main()
