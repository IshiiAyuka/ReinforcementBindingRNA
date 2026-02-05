#!/usr/bin/env python3
import math
from collections import Counter

import pandas as pd

#INPUT_CSV = "DecoderOnly_RNAcompete.csv"
INPUT_CSV = "BAnG_RNAcompete.csv"
#INPUT_CSV = "All_MFE_RNAcompete.csv"
KMER_LENGTH = 7


def shannon_entropy_kmers(seqs, k):
    counts = Counter()
    total = 0
    for seq in seqs:
        if seq is None:
            continue
        s = str(seq).strip()
        if len(s) < k:
            continue
        for i in range(len(s) - k + 1):
            counts[s[i : i + k]] += 1
            total += 1
    if total == 0:
        return 0.0, 0, 0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p, 2)
    return h, total, len(counts)


def main():
    df = pd.read_csv(INPUT_CSV)
    if "id" not in df.columns or "sequence" not in df.columns:
        raise SystemExit("Input CSV must contain 'id' and 'sequence' columns")

    print(f"id,shannon_entropy_k{KMER_LENGTH},Total_kmers,Unique_kmers")
    for id_, group in df.groupby("id"):
        h, total, uniq = shannon_entropy_kmers(group["sequence"], KMER_LENGTH)
        print(f"{id_},{h},{total},{uniq}")


if __name__ == "__main__":
    main()
