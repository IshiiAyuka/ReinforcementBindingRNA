import pandas as pd
from tqdm import tqdm

# === ファイル読み込み ===
uniprot_df = pd.read_csv("uniprot_sequences.csv")
ppi_df = pd.read_csv("ppi3d.csv")

# === PPI3Dの辞書を作成（sequence -> subunit_1）===
ppi_seq_to_subunit = dict(zip(ppi_df["s1_sequence"], ppi_df["subunit_1"]))

# === 一致結果を格納するリスト ===
matched = []

# === 1つずつUniProt配列と比較 ===
for idx, row in tqdm(uniprot_df.iterrows(), total=len(uniprot_df), desc="配列比較中"):
    sequence = row["sequence"]
    if sequence in ppi_seq_to_subunit:
        matched_subunit = ppi_seq_to_subunit[sequence]
        print(f"一致: {row['protein_name']} / UniProt ID: {row['uniprot_id']} / subunit_1: {matched_subunit}", flush=True)
        matched.append({
            "protein_name": row["protein_name"],
            "uniprot_id": row["uniprot_id"],
            "subunit_1": matched_subunit
        })

# === 結果をDataFrameにして保存 ===
if matched:
    matched_df = pd.DataFrame(matched)
    matched_df.to_csv("matched_protein_subunits.csv", index=False)
    print(f"\n一致結果を保存しました: matched_protein_subunits.csv", flush=True)
else:
    print("\n一致する配列は見つかりませんでした。", flush=True)
