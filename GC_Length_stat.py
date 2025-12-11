from pathlib import Path
import csv

csv_path = Path("/home/slab/ishiiayuka/M2/deepclip/generated_rna/generated_rna_RNCMPT_t30_150M_AR_1206.csv")

n = 0
mean_gc = m2_gc = 0.0
mean_len = m2_len = 0.0

with csv_path.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        seq = row["sequence"].strip()
        if not seq:
            continue

        length = float(len(seq))
        gc_frac = (seq.count("G") + seq.count("C")) / length

        n += 1
        # Welford: GC
        delta_gc = gc_frac - mean_gc
        mean_gc += delta_gc / n
        m2_gc += delta_gc * (gc_frac - mean_gc)
        # Welford: length
        delta_len = length - mean_len
        mean_len += delta_len / n
        m2_len += delta_len * (length - mean_len)

if n < 2:
    std_gc = std_len = 0.0
else:
    std_gc = (m2_gc / (n - 1)) ** 0.5
    std_len = (m2_len / (n - 1)) ** 0.5

print(f"Total sequences: {n}")
print(f"GC content mean: {mean_gc:.6f}")
print(f"GC content std dev: {std_gc:.6f}")
print(f"Sequence length mean: {mean_len:.2f}")
print(f"Sequence length std dev: {std_len:.2f}")
