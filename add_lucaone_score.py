#!/usr/bin/env python3
# encoding: utf-8

from pathlib import Path
import multiprocessing as mp
import os
import traceback

import pandas as pd
from tqdm import tqdm

from LucaOneTasks.src.predict_v1 import run as predict_run


INPUT_CSV = "deepclip_result/DecoderOnly_RNAcompete_with_score.csv"
OUTPUT_CSV = "DecoderOnly_RNAcompete_with_score_lucaone.csv"
ID_COL = "id"
SEQ_COL = "sequence"
SCORE_COL = "LucaOne_Score"
BATCH_SIZE = 8
GPU_ID = 3
GPU_IDS = [0, 1, 2, 3]
BALANCE_BY_LENGTH = True

# LucaOne model settings (match reinforce_swissprot_AR_offtarget_allreward.py).
MODEL_PATH = "LucaOneTasks/"
LLM_TRUNCATION_SEQ_LENGTH = 100
DATASET_NAME = "ncRPI"
DATASET_TYPE = "gene_protein"
TASK_TYPE = "binary_class"
TASK_LEVEL_TYPE = "seq_level"
MODEL_TYPE = "lucappi2"
INPUT_TYPE = "matrix"
INPUT_MODE = "pair"
TIME_STR = "20240404105148"
STEP = "716380"
THRESHOLD = 0.5
TOPK = None
EMB_DIR = None
MATRIX_EMBEDDING_EXISTS = False

# Pair-mode protein settings: either set PROTEIN_SEQ or use CSV columns.
PROTEIN_SEQ = None
PROTEIN_ID_COL = None
PROTEIN_COL = None
PROTEIN_MAP_CSV = "RNAcompete.csv"
PROTEIN_MAP_KEY_COL = "file_name"
PROTEIN_MAP_SEQ_COL = "sequence"


def normalize_protein_key(value):
    key = str(value)
    if key.endswith(".pkl"):
        key = key[:-4]
    return key


def load_protein_map(path, key_col, seq_col):
    protein_df = pd.read_csv(path)
    if key_col not in protein_df.columns:
        raise KeyError(f"Missing protein key column: {key_col}")
    if seq_col not in protein_df.columns:
        raise KeyError(f"Missing protein sequence column: {seq_col}")

    mapping = {}
    for _, row in protein_df.iterrows():
        key = normalize_protein_key(row[key_col])
        mapping[key] = str(row[seq_col]).upper()
    return mapping


def prepare_protein_lookup(df, id_col, seq_col):
    if id_col not in df.columns:
        raise KeyError(f"Missing id column: {id_col}")
    if seq_col not in df.columns:
        raise KeyError(f"Missing sequence column: {seq_col}")

    if INPUT_MODE != "pair":
        return None
    if PROTEIN_SEQ is not None:
        return None
    if PROTEIN_ID_COL and PROTEIN_COL:
        if PROTEIN_ID_COL not in df.columns:
            raise KeyError(f"Missing protein id column: {PROTEIN_ID_COL}")
        if PROTEIN_COL not in df.columns:
            raise KeyError(f"Missing protein column: {PROTEIN_COL}")
        return None
    return load_protein_map(
        PROTEIN_MAP_CSV,
        PROTEIN_MAP_KEY_COL,
        PROTEIN_MAP_SEQ_COL,
    )


def build_sequence_list(df, protein_map):
    sequences = []
    for idx, row in enumerate(df.itertuples(index=False)):
        seq_id = str(getattr(row, ID_COL))
        seq = str(getattr(row, SEQ_COL)).upper()
        if INPUT_MODE == "pair":
            if PROTEIN_SEQ is None:
                if PROTEIN_ID_COL and PROTEIN_COL:
                    prot_id = str(getattr(row, PROTEIN_ID_COL))
                    prot_seq = str(getattr(row, PROTEIN_COL)).upper()
                else:
                    prot_id = seq_id
                    prot_key = normalize_protein_key(seq_id)
                    if prot_key not in protein_map:
                        raise KeyError(f"Missing protein sequence for key: {prot_key}")
                    prot_seq = protein_map[prot_key]
            else:
                prot_id = "protein_fixed"
                prot_seq = str(PROTEIN_SEQ).upper()
            total_len = len(seq) + len(prot_seq)
            sequences.append((idx, [seq_id, prot_id, "gene", "prot", seq, prot_seq], total_len))
        else:
            sequences.append((idx, [seq_id, "gene", seq], len(seq)))
    return sequences


def chunk_list(items, chunk_size):
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def maybe_balance_sequences(sequences):
    if not BALANCE_BY_LENGTH:
        return sequences
    return sorted(sequences, key=lambda x: x[2], reverse=True)


def worker_run(gpu_id, task_queue, out_queue):
    try:
        out_queue.put(("start", gpu_id, os.getpid()))
        while True:
            chunk = task_queue.get()
            if chunk is None:
                break
            seq_batch = [item[1] for item in chunk]
            results = predict_run(
                seq_batch,
                llm_truncation_seq_length=LLM_TRUNCATION_SEQ_LENGTH,
                model_path=MODEL_PATH,
                dataset_name=DATASET_NAME,
                dataset_type=DATASET_TYPE,
                task_type=TASK_TYPE,
                task_level_type=TASK_LEVEL_TYPE,
                model_type=MODEL_TYPE,
                input_type=INPUT_TYPE,
                input_mode=INPUT_MODE,
                time_str=TIME_STR,
                step=str(STEP),
                gpu_id=gpu_id,
                threshold=THRESHOLD,
                topk=TOPK,
                emb_dir=EMB_DIR,
                matrix_embedding_exists=MATRIX_EMBEDDING_EXISTS,
            )
            for (idx, _, _), row in zip(chunk, results):
                out_queue.put(("result", idx, extract_score(TASK_TYPE, row)))
            out_queue.put(("progress", gpu_id, len(chunk)))
        out_queue.put(("done", gpu_id))
    except Exception as exc:
        out_queue.put(("error", gpu_id, repr(exc), traceback.format_exc()))


def run_single_gpu(sequences, total):
    scores = [None] * total
    progress = tqdm(total=total, unit="seq", desc="LucaOne scoring")
    for i in range(0, len(sequences), BATCH_SIZE):
        chunk = sequences[i : i + BATCH_SIZE]
        seq_batch = [item[1] for item in chunk]
        results = predict_run(
            seq_batch,
            llm_truncation_seq_length=LLM_TRUNCATION_SEQ_LENGTH,
            model_path=MODEL_PATH,
            dataset_name=DATASET_NAME,
            dataset_type=DATASET_TYPE,
            task_type=TASK_TYPE,
            task_level_type=TASK_LEVEL_TYPE,
            model_type=MODEL_TYPE,
            input_type=INPUT_TYPE,
            input_mode=INPUT_MODE,
            time_str=TIME_STR,
            step=str(STEP),
            gpu_id=GPU_ID,
            threshold=THRESHOLD,
            topk=TOPK,
            emb_dir=EMB_DIR,
            matrix_embedding_exists=MATRIX_EMBEDDING_EXISTS,
        )
        for (idx, _, _), row in zip(chunk, results):
            scores[idx] = extract_score(TASK_TYPE, row)
        progress.update(len(chunk))
    progress.close()
    return scores


def run_multi_gpu(sequences, total):
    gpu_ids = GPU_IDS
    if not gpu_ids:
        raise ValueError("GPU_IDS is empty for multi-GPU run.")

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    task_queue = ctx.Queue()
    procs = []
    for gpu_id in gpu_ids:
        p = ctx.Process(target=worker_run, args=(gpu_id, task_queue, out_queue))
        p.start()
        procs.append(p)

    chunks = chunk_list(sequences, BATCH_SIZE)
    for chunk in chunks:
        task_queue.put(chunk)
    for _ in gpu_ids:
        task_queue.put(None)

    scores = [None] * total
    progress = tqdm(total=total, unit="seq", desc="LucaOne scoring")
    done = 0
    while done < len(procs):
        msg = out_queue.get()
        if msg[0] == "start":
            _, gpu_id, pid = msg
            print(f"[info] worker started gpu_id={gpu_id} pid={pid}", flush=True)
        elif msg[0] == "result":
            _, idx, score = msg
            scores[idx] = score
        elif msg[0] == "progress":
            progress.update(msg[2])
        elif msg[0] == "error":
            _, gpu_id, err, tb = msg
            for p in procs:
                if p.is_alive():
                    p.terminate()
            progress.close()
            raise RuntimeError(f"worker error gpu_id={gpu_id}: {err}\n{tb}")
        elif msg[0] == "done":
            done += 1
    progress.close()
    for p in procs:
        p.join()
    return scores


def extract_score(task_type, result_row):
    # result_row for pair input usually: [id_a, id_b, seq_a, seq_b, score, label]
    # result_row for single input usually: [seq_id, seq, score, label]
    score_index = 4 if len(result_row) >= 6 else 2
    score = result_row[score_index]
    if isinstance(score, list):
        return max(score) if score else 0.0
    return float(score)


def main():
    input_path = Path(INPUT_CSV)
    output_path = Path(OUTPUT_CSV)

    df = pd.read_csv(input_path)
    protein_map = prepare_protein_lookup(df, ID_COL, SEQ_COL)

    sequences = build_sequence_list(df, protein_map)
    total = len(df)
    if GPU_IDS:
        print(
            f"[info] rows={total} batch_size={BATCH_SIZE} gpu_ids={GPU_IDS}",
            flush=True,
        )
        scores = run_multi_gpu(maybe_balance_sequences(sequences), total)
    else:
        print(
            f"[info] rows={total} batch_size={BATCH_SIZE} gpu_id={GPU_ID}",
            flush=True,
        )
        scores = run_single_gpu(sequences, total)

    if len(scores) != len(df):
        raise RuntimeError(
            f"Score count mismatch: {len(scores)} scores for {len(df)} rows."
        )

    df[SCORE_COL] = scores
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
