import os
import gzip
import pickle
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1
from tqdm import tqdm

# 入力と出力のパス
input_dir = "./data"  # cif.gzファイルがあるディレクトリ
output_pkl = "complex_sequences.pkl"

# アミノ酸3文字コードのリスト
amino_acids = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL"
}

rna_bases = {"A", "U", "C", "G", "I"}

# 構造パーサーの準備
parser = MMCIFParser(QUIET=True)

# 出力用の辞書
complex_data = {}

# ディレクトリ内のcif.gzファイルを処理
for file in tqdm(sorted(os.listdir(input_dir))):
    if not file.endswith(".cif.gz"):
        continue

    file_path = os.path.join(input_dir, file)
    complex_id = file.replace(".cif.gz", "")
    complex_data[complex_id] = {
        "proteins": {},  # {"chain_id": aa_seq}
        "rnas": {}       # {"chain_id": rna_seq}
    }

    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as handle:
            structure = parser.get_structure(complex_id, handle)
            model = next(structure.get_models())

            for chain in model:
                chain_id = chain.id
                aa_seq = ""
                rna_seq = ""

                for residue in chain:
                    resname = residue.get_resname().upper()

                    if resname in amino_acids:
                        try:
                            aa_seq += seq1(resname)
                        except Exception:
                            continue
                    elif resname in rna_bases:
                        rna_seq += resname

                if len(aa_seq) > 0:
                    complex_data[complex_id]["proteins"][chain_id] = aa_seq
                if len(rna_seq) > 0:
                    complex_data[complex_id]["rnas"][chain_id] = rna_seq

    except Exception as e:
        print(f"{file} の解析中にエラー: {e}")

# 出力ファイルに保存
with open(output_pkl, "wb") as f:
    pickle.dump(complex_data, f)

print(f"保存完了: {output_pkl}")
