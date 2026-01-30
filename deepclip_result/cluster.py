import os
import shutil
import subprocess
import tempfile

import pandas as pd


CSV_PATH = "/home/slab/ishiiayuka/M2/deepclip_result/All_MFE_RNAcompete.csv"
#CSV_PATH = "/home/slab/ishiiayuka/M2/deepclip_result/BAnG_RNAcompete.csv"
IDENTITY = 0.1



def write_fasta(seqs, path):
    with open(path, "w") as f:
        for i, seq in enumerate(seqs, 1):
            f.write(f">seq{i}\n{seq}\n")


def count_clusters_uc(uc_path):
    clusters = set()
    with open(uc_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            clusters.add(parts[1])
    return len(clusters)


def run_vsearch(fasta_in, uc_out, identity):
    if shutil.which("vsearch") is None:
        raise FileNotFoundError("vsearch not found in PATH.")
    cmd = [
        "vsearch",
        "--cluster_fast",
        fasta_in,
        "--id",
        str(identity),
        "--uc",
        uc_out,
        "--quiet",
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        if e.stderr:
            print(e.stderr.strip())
        raise


def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.astype(str).str.strip()

    if "id" not in df.columns or "sequence" not in df.columns:
        raise KeyError("CSV must contain 'id' and 'sequence' columns.")

    df = df[["id", "sequence"]].dropna()

    for protein_id, group in df.groupby("id"):
        seqs = [str(s).strip() for s in group["sequence"] if str(s).strip()]
        if not seqs:
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_in = os.path.join(tmpdir, "input.fasta")
            uc_out = os.path.join(tmpdir, "vsearch.uc")
            write_fasta(seqs, fasta_in)
            run_vsearch(fasta_in, uc_out, IDENTITY)
            clusters = count_clusters_uc(uc_out)

        total = len(seqs)
        ratio = clusters / total if total else 0.0
        print(f"{protein_id},{clusters},{total},{ratio:.4f}")


if __name__ == "__main__":
    main()
