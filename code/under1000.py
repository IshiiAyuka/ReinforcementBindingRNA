import os
import shutil
import gzip
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1
from tqdm import tqdm

# フォルダのパス
input_dir = "data"
output_dir = "under500data"
os.makedirs(output_dir, exist_ok=True)

# アミノ酸の3文字→1文字変換
def get_protein_sequences(cif_file):
    try:
        parser = MMCIFParser(QUIET=True)
        # gzファイルの場合
        if cif_file.endswith(".gz"):
            with gzip.open(cif_file, "rt") as handle:
                structure = parser.get_structure("structure", handle)
        else:
            structure = parser.get_structure("structure", cif_file)

        amino_acids = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
            "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
            "THR", "TRP", "TYR", "VAL"
        }

        sequences = []

        for model in structure:
            for chain in model:
                aa_list = []
                for residue in chain:
                    if residue.get_resname() in amino_acids:
                        aa_list.append(seq1(residue.get_resname()))
                if aa_list:
                    sequences.append("".join(aa_list))
        return sequences
    except Exception as e:
        print(f"エラー ({cif_file}): {e}")
        return []

# 全ファイルを確認して、条件を満たすものだけコピー
all_files = [f for f in os.listdir(input_dir) if f.endswith(".cif") or f.endswith(".cif.gz")]

for file_name in tqdm(all_files, desc="処理中"):
    file_path = os.path.join(input_dir, file_name)
    sequences = get_protein_sequences(file_path)

    if sequences and all(len(seq) <= 500 for seq in sequences):
        shutil.copy(file_path, os.path.join(output_dir, file_name))
        lengths = [len(seq) for seq in sequences]
        tqdm.write(f"コピー済み: {file_name}（各配列の長さ: {lengths}）")
