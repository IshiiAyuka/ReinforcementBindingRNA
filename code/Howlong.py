import os
import gzip
import glob
import matplotlib.pyplot as plt
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1
from tqdm import tqdm

# アミノ酸3文字表記セット
amino_acids = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL"
}

def extract_lengths_from_cif(cif_path):
    parser = MMCIFParser(QUIET=True)
    aa_lengths = []
    rna_lengths = []

    try:
        with gzip.open(cif_path, 'rt', encoding='utf-8') as handle:
            structure = parser.get_structure("structure", handle)

            for model in structure:
                for chain in model:
                    aa_seq = ""
                    rna_seq = ""
                    for residue in chain:
                        resname = residue.get_resname()
                        if resname in amino_acids:
                            try:
                                aa_seq += seq1(resname)
                            except:
                                continue
                        elif resname in ["A", "U", "C", "G", "I"]:
                            rna_seq += resname
                    if len(aa_seq) > 0:
                        aa_lengths.append(len(aa_seq))
                    if len(rna_seq) > 0:
                        rna_lengths.append(len(rna_seq))

    except Exception as e:
        print(f"Error reading {cif_path}: {e}")

    return aa_lengths, rna_lengths

# 対象ディレクトリ
cif_dir = "./filtered_data"
cif_files = glob.glob(os.path.join(cif_dir, "*.cif.gz"))

all_aa_lengths = []
all_rna_lengths = []

print(f"処理対象ファイル数: {len(cif_files)}")

# ファイルごとに長さを抽出
for cif_file in tqdm(cif_files, desc="ファイル処理中", unit="ファイル"):
    aa_lengths, rna_lengths = extract_lengths_from_cif(cif_file)
    all_aa_lengths.extend(aa_lengths)
    all_rna_lengths.extend(rna_lengths)

# 最短・最長を出力
if all_aa_lengths:
    print(f"アミノ酸配列長：最短 = {min(all_aa_lengths)}, 最長 = {max(all_aa_lengths)}")
else:
    print("アミノ酸配列が見つかりませんでした。")

if all_rna_lengths:
    print(f"RNA配列長：最短 = {min(all_rna_lengths)}, 最長 = {max(all_rna_lengths)}")
else:
    print("RNA配列が見つかりませんでした。")

# アミノ酸長ヒストグラム
plt.figure(figsize=(10, 6))
plt.hist(all_aa_lengths, bins=50, edgecolor='black')
plt.title("amino_acid")
plt.xlabel("length")
plt.ylabel("frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("filtered_amino_acid_length_histogram.png")
print("アミノ酸配列長ヒストグラムを保存しました。")

# RNA長ヒストグラム
plt.figure(figsize=(10, 6))
plt.hist(all_rna_lengths, bins=50, edgecolor='black')
plt.title("RNA")
plt.xlabel("length")
plt.ylabel("frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("filtered_rna_length_histogram.png")
print("RNA配列長ヒストグラムを保存しました。")
