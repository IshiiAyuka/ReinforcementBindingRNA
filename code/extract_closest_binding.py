import pandas as pd
import numpy as np
import os
from Bio.PDB import MMCIFParser
from tqdm import tqdm
import gzip
import csv

def get_residue_centroid(residue):
    """残基の重心（Hydrogen原子は除く）"""
    coords = [atom.coord for atom in residue if atom.element != 'H']
    return np.mean(coords, axis=0) if coords else None

def find_closest_residue_pair(cif_gz_path, protein_chain_id, rna_chain_id):
    parser = MMCIFParser(QUIET=True)

    # gzip解凍して、Biopythonに渡す
    with gzip.open(cif_gz_path, "rt") as gz_file:  # "rt" = read text mode
        structure = parser.get_structure("complex", gz_file)

    model = structure[0]
    min_distance = float("inf")
    closest_pair = (None, None)

    try:
        protein_chain = model[protein_chain_id]
        rna_chain = model[rna_chain_id]
    except KeyError:
        raise ValueError(f"指定チェーンが見つかりません: {protein_chain_id}, {rna_chain_id}")

    for res_p in protein_chain:
        centroid_p = get_residue_centroid(res_p)
        if centroid_p is None:
            continue
        for res_r in rna_chain:
            centroid_r = get_residue_centroid(res_r)
            if centroid_r is None:
                continue
            dist = np.linalg.norm(centroid_p - centroid_r)
            if dist < min_distance:
                min_distance = dist
                closest_pair = (res_p.id[1], res_r.id[1])

    return closest_pair, min_distance

def process_ppi3d_csv(csv_path, data_dir="./data", output_path="binding_result.csv"):
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    with open(output_path, "w", newline="") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["pdb_id", "protein_chain", "rna_chain", "closest_protein_residue", "closest_rna_residue", "distance"])

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="結合残基探索中", ncols=100):
            try:
                pdb_id = str(row["pdb_id"]).lower()
                protein_chain = row["subunit_1"].split("_")[1]
                rna_chain = row["subunit_2"].split("_")[1]
                cif_file = os.path.join(data_dir, f"{pdb_id}.cif.gz")

                if not os.path.exists(cif_file):
                    print(f"[{pdb_id}] ファイルが見つかりません: {cif_file}")
                    continue

                (res_p, res_r), distance = find_closest_residue_pair(cif_file, protein_chain, rna_chain)

                row_data = [pdb_id.upper(), protein_chain, rna_chain, res_p, res_r, round(distance, 2)]
                writer.writerow(row_data)
                print(",".join(map(str, row_data)))

            except Exception as e:
                error_data = [row.get("pdb_id", "ERROR"), "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"]
                writer.writerow(error_data)
                print( ",".join(map(str, error_data)), f"理由: {e}")


    print(f"結果を {output_path} に保存しました。")

# -------------------------
# 実行ブロック（ここに記述）
# -------------------------
if __name__ == "__main__":
    process_ppi3d_csv("ppi3d.csv", data_dir="./data", output_path="binding_result.csv")
