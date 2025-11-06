#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, re, shutil, tempfile, subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def which(cmds):
    for c in cmds:
        p = shutil.which(c)
        if p: return p
    return None

def ensure_datapath():
    dp = os.environ.get("DATAPATH", "")
    if dp and Path(dp).exists(): return
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    guess = Path(conda_prefix) / "share" / "rnastructure" / "data_tables"
    if conda_prefix and guess.exists():
        os.environ["DATAPATH"] = str(guess)

def sanitize_rna(seq: str) -> str:
    if pd.isna(seq): return ""
    s = str(seq).strip().upper().replace("T","U")
    s = re.sub(r"^>.*\n","", s)
    s = re.sub(r"\s+","", s)
    return s

def write_fasta(path: Path, name: str, seq: str):
    with open(path, "w") as f:
        f.write(f">{name}\n{seq}\n")

def kmers(s: str, k: int):
    return {s[i:i+k] for i in range(len(s)-k+1)} if len(s) >= k else set()

def best_kmer_window(true_seq: str, pred_seq: str, k: int = 6, flank: int | None = None):
    if k is None or k < 1 or len(true_seq) < k or len(pred_seq) < k: return None
    ts = kmers(true_seq, k)
    hit = next((i for i in range(len(pred_seq)-k+1) if pred_seq[i:i+k] in ts), None)
    if hit is None: return None
    if flank is None: flank = max(150, len(true_seq))
    s = max(0, hit - flank); e = min(len(pred_seq), hit + k + flank)
    return pred_seq[s:e]

def make_dynconf(conf_path: Path, in1: Path, in2: Path, out1: Path, out2: Path, aout: Path, threads: int):
    lines = [
        f"inseq1 = {in1}",
        f"inseq2 = {in2}",
        f"outct = {out1}",
        f"outct2 = {out2}",
        f"aout = {aout}",
        "local = 1",
        "optimal_only = 1",
        f"num_processors = {max(1, int(threads))}",
        "percent = 20",
        "bpwin = 2",
        "awin = 1",
    ]
    conf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def parse_align_ali(path: Path) -> int:
    if not path.exists(): return 0
    pairs = 0
    with open(path, "r") as f:
        for line in f:
            m = re.match(r"\s*(-?\d+)\s+(-?\d+)", line)
            if not m: continue
            i, j = int(m.group(1)), int(m.group(2))
            if i == -1 and j == -1: break
            if i > 0 and j > 0: pairs += 1
    return pairs

def main():
    ap = argparse.ArgumentParser(description="CSVの true_rna_seq / pred_rna_seq を Dynalign II で逐次評価（TSV出力, スキップ/タイムアウト対応）")
    ap.add_argument("--csv", required=True, help="入力CSV（列: pdb_id,true_rna_seq,pred_rna_seq）")
    ap.add_argument("--threads", type=int, default=8, help="dynalign_ii-smp の並列数")
    ap.add_argument("--kmer", type=int, default=3, help="pred窓切出し用k（0で無効）")
    ap.add_argument("--flank", type=int, default=0, help="ヒット周辺片側フランク長（0で自動: max(150,len_true)）")
    ap.add_argument("--timeout", type=int, default=180, help="1行あたりタイムアウト秒（0で無効）")
    ap.add_argument("--skip_if_no_hit", action="store_true",
                    help="k-merヒットが無く、かつ pred/true 長比が --max_ratio 超なら Dynalign をスキップ")
    ap.add_argument("--max_ratio", type=float, default=4.0,
                    help="pred/true 長比の上限（--skip_if_no_hit と併用）")
    args = ap.parse_args()

    exe = which(["dynalign_ii-smp", "dynalign_ii"])
    if not exe:
        print("ERROR: dynalign_ii-smp / dynalign_ii が見つかりません。conda 環境と PATH を確認してください。", file=sys.stderr)
        sys.exit(2)

    ensure_datapath()

    # CSV
    try:
        df = pd.read_csv(args.csv, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(args.csv, sep="\t")

    cols = {c.strip().lower(): c for c in df.columns}
    for k in ["pdb_id","true_rna_seq","pred_rna_seq"]:
        if k not in cols:
            print(f"ERROR: 列 {k} が見つかりません。実列: {list(df.columns)}", file=sys.stderr)
            sys.exit(3)

    # TSVヘッダはstdoutへ、進捗はstderrへ
    print("pdb_id\tlen_true\tlen_pred\tdynalign_pairs\ttrue_coverage\tstatus\tnote", flush=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Dynalign", unit="row", file=sys.stderr):
        pdb_id = str(row[cols["pdb_id"]])
        true_seq = sanitize_rna(row[cols["true_rna_seq"]])
        pred_seq = sanitize_rna(row[cols["pred_rna_seq"]])

        status, note = "ok", ""
        pairs = 0
        len_true, len_pred = len(true_seq), len(pred_seq)

        if not true_seq or not pred_seq:
            status, note = "empty_seq", "true または pred が空"
            print(f"{pdb_id}\t{len_true}\t{len_pred}\t{pairs}\t{0.0:.6f}\t{status}\t{note}", flush=True)
            continue

        # 窓切り出し & ヒットなければスキップ可
        pred_for_align = pred_seq
        did_window = False
        if args.kmer and len_pred > len_true * args.max_ratio:
            flank = args.flank if args.flank > 0 else max(150, len_true)
            win = best_kmer_window(true_seq, pred_seq, k=args.kmer, flank=flank)
            if win:
                pred_for_align = win
                did_window = True
                note = f"windowed_pred({len_pred}->{len(pred_for_align)})"
            elif args.skip_if_no_hit:
                status = "skipped_no_kmer_hit"
                note = f"no_kmer_hit; ratio={len_pred/len_true:.2f}"
                print(f"{pdb_id}\t{len_true}\t{len_pred}\t{pairs}\t{0.0:.6f}\t{status}\t{note}", flush=True)
                continue
            else:
                note = "no_kmer_hit; used_full_pred"

        try:
            with tempfile.TemporaryDirectory(prefix="dyn_") as td:
                td = Path(td)
                f_true, f_pred = td/"true.fa", td/"pred.fa"
                write_fasta(f_true, "true", true_seq)
                write_fasta(f_pred, "pred", pred_for_align)

                conf, out1, out2, aout = td/"dyn.conf", td/"true.ct", td/"pred.ct", td/"align.ali"
                make_dynconf(conf, f_true, f_pred, out1, out2, aout, threads=args.threads)

                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(max(1, int(args.threads)))
                # （好みで）環境によっては OMP_DYNAMIC=FALSE が安定
                env["OMP_DYNAMIC"] = "FALSE"

                kwargs = dict(capture_output=True, text=True, cwd=str(td), env=env)
                if args.timeout and args.timeout > 0:
                    kwargs["timeout"] = args.timeout

                proc = subprocess.run([exe, str(conf)], **kwargs)
                if proc.returncode != 0:
                    status = "dynalign_error"
                    head = (proc.stderr or proc.stdout or "").strip().splitlines()[:2]
                    note = (note + "; " if note else "") + f"ret={proc.returncode}" + (" | " + " | ".join(head) if head else "")
                else:
                    pairs = parse_align_ali(aout)
                    if pairs == 0:
                        status = "no_alignment"

        except subprocess.TimeoutExpired as e:
            status = "timeout"
            note = (note + "; " if note else "") + f"dynalign timeout ({e.timeout}s)"

        true_cov = (pairs / len_true) if len_true else 0.0
        print(f"{pdb_id}\t{len_true}\t{len_pred}\t{pairs}\t{true_cov:.6f}\t{status}\t{note}", flush=True)

if __name__ == "__main__":
    main()
