import pandas as pd
from tqdm import tqdm

def extract_trimmed_rna(ppi3d_csv, binding_csv, output_path="binding_trimmed.csv", flank=37):
    # データ読み込み
    df_ppi  = pd.read_csv(ppi3d_csv, low_memory=False)
    df_bind = pd.read_csv(binding_csv, low_memory=False)

    # カラム名クリーニング
    df_ppi.columns  = df_ppi.columns.str.strip().str.lower()
    df_bind.columns = df_bind.columns.str.strip().str.lower()

    # subunit_1/2 を「pdb lowercase + '_' + chain uppercase」の形式に整形
    def normalize_subunit(s):
        pdb, chain = s.split("_")
        return pdb.lower() + "_" + chain.upper()

    df_ppi["subunit_1"] = df_ppi["subunit_1"].apply(normalize_subunit)
    df_ppi["subunit_2"] = df_ppi["subunit_2"].apply(normalize_subunit)

    # トリミング結果用カラム
    df_ppi["trimmed_rna_sequence"] = pd.NA

    for _, row in tqdm(df_bind.iterrows(), total=len(df_bind), desc="RNA trimming", ncols=80):
        pdb        = str(row["pdb_id"]).lower()
        prot_chain = str(row["protein_chain_id"]).upper()
        rna_chain  = str(row["rna_chain_id"]).upper()
        try:
            rna_pos = int(row["closest_rna_id"])
        except:
            continue

        key1 = f"{pdb}_{prot_chain}"
        key2 = f"{pdb}_{rna_chain}"

        mask = (df_ppi["subunit_1"] == key1) & (df_ppi["subunit_2"] == key2)
        if not mask.any():
            continue

        idx     = df_ppi[mask].index[0]
        rna_seq = df_ppi.at[idx, "s2_sequence"]
        start   = max(0, rna_pos - flank - 1)
        end     = min(len(rna_seq), rna_pos + flank)
        trimmed = rna_seq[start:end]

        df_ppi.at[idx, "trimmed_rna_sequence"] = trimmed

    # trimmed_rna_sequence が入っている行だけを出力
    df_out = df_ppi[df_ppi["trimmed_rna_sequence"].notna()]
    df_out.to_csv(output_path, index=False)
    print(f"{len(df_out)} rows saved to {output_path}")

if __name__ == "__main__":
    extract_trimmed_rna("ppi3d.csv", "ppi3d_under75_extract.csv", "binding_trimmed.csv")
