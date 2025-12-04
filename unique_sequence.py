import pandas as pd

in_csv = "/home/slab/ishiiayuka/M2/deepclip/generated_rna/generated_rna_RNCMPT_t30_150M_AR_1202_4.csv"
df = pd.read_csv(in_csv, dtype=str)

df["id"] = df["id"].str.strip()
df["sequence"] = df["sequence"].str.strip()
df = df.dropna(subset=["id", "sequence"])
df = df[df["sequence"] != ""]
print("id,unique_percent")

for _id, g in df.groupby("id"):
    total = len(g)
    unique = g["sequence"].nunique()
    unique_percent = unique / total if total else 0.0
    print(f"{_id},{unique_percent:.2f}")
