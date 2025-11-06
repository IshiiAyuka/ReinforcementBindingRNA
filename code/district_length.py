import os
import gzip
import shutil
from tqdm import tqdm
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1

# 入出力フォルダの指定
input_dir = "data"
output_dir = "filtered_data"
os.makedirs(output_dir, exist_ok=True)

# アミノ酸名3文字→1文字への変換関数
def convert_resname_to_aa(resname):
    try:
        return seq1(resname)
    except:
        return None

# ファイルごとのフィルタ関数
def is_valid_complex(file_path):
    parser = MMCIFParser(QUIET=True)
    aa_lengths = []
    rna_lengths = []

    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as handle:
            structure = parser.get_structure("complex", handle)

        for model in structure:
            for chain in model:
                aa_seq = ""
                rna_seq = ""
                for residue in chain:
                    resname = residue.get_resname()
                    if resname in ["A", "U", "C", "G", "I"]:  # RNA塩基
                        rna_seq += resname
                    else:
                        aa = convert_resname_to_aa(resname)
                        if aa:
                            aa_seq += aa
                if aa_seq:
                    aa_lengths.append(len(aa_seq))
                if rna_seq:
                    rna_lengths.append(len(rna_seq))

        # 結果の出力
        print(f"{os.path.basename(file_path)}")
        print(f"   - アミノ酸配列長: {aa_lengths}")
        print(f"   - RNA配列長: {rna_lengths}")

        # すべての配列が500以下であるかチェック
        if all(l <= 500 for l in aa_lengths + rna_lengths):
            return True

    except Exception as e:
        print(f"エラー ({file_path}): {e}")

    return False

# tqdmを使って進捗を表示
file_list = [f for f in os.listdir(input_dir) if f.endswith(".cif.gz")]

for file_name in tqdm(file_list, desc="Filtering CIF files"):
    full_path = os.path.join(input_dir, file_name)
    if is_valid_complex(full_path):
        shutil.copy(full_path, os.path.join(output_dir, file_name))
        print(f"コピー: {file_name}\n")
    else:
        print(f"⏭スキップ: {file_name}\n")
