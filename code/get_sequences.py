import os
import glob
import gzip
import pickle
from tqdm import tqdm
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1

# 配列抽出関数
def extract_sequences_from_cif(file_path):
    parser = MMCIFParser(QUIET=True)
    aa_seq, rna_seq = "", ""

    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as handle:
            structure = parser.get_structure("complex", handle)

        model_obj = next(structure.get_models())
        chain = next(model_obj.get_chains())

        for residue in chain:
            resname = residue.get_resname()
            if resname in ["A", "U", "C", "G", "I"]:
                rna_seq += resname
            else:
                try:
                    aa_seq += seq1(resname)
                except Exception:
                    continue

    except Exception as e:
        print(f"エラー: {file_path} - {e}")
        return "", ""

    return aa_seq, rna_seq

# データセット作成関数（配列のみ）
def build_sequence_dataset(cif_folder, output_file):
    dataset = []
    files = glob.glob(os.path.join(cif_folder, "*.cif.gz"))

    for path in tqdm(files, desc="配列抽出中"):
        aa_seq, rna_seq = extract_sequences_from_cif(path)
        dataset.append((aa_seq, rna_seq))

    print(f"有効なデータ数: {len(dataset)} 件")
    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)
    print(f"配列データを保存しました: {output_file}")

# 実行
if __name__ == "__main__":
    build_sequence_dataset(cif_folder="data", output_file="sequences.pkl")
