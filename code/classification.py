import os
import gzip
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

def clean_and_split_keywords(raw_keyword):
    if raw_keyword is None:
        return ["Unknown"]
    keyword = raw_keyword.replace("'", "").replace('"', '')
    keyword = re.sub(r"(/(RNA|DNA|rna|dna))+", "", keyword)
    keyword = keyword.strip()
    split_keywords = [k.strip() for k in keyword.split("/") if k.strip()]
    return split_keywords if split_keywords else ["Unknown"]

# データ格納ディレクトリ
cif_dir = "./data"  
cif_files = [f for f in os.listdir(cif_dir) if f.endswith(".cif.gz")]

def extract_struct_keywords(cif_path):
    try:
        with gzip.open(cif_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.startswith("_struct_keywords.pdbx_keywords"):
                    raw = line.strip().split(" ", 1)[-1].strip()
                    return clean_and_split_keywords(raw)
        return ["Unknown"]
    except Exception as e:
        print(f"{cif_path} の読み込みエラー: {e}")
        return ["読み込みエラー"]

# キーワード収集
keyword_counter = Counter()
for fname in tqdm(cif_files, desc="複合体キーワードの解析中"):
    pdb_id = os.path.splitext(os.path.splitext(fname)[0])[0]
    cif_path = os.path.join(cif_dir, fname)
    keywords = extract_struct_keywords(cif_path)  # listで返る
    print(f"{pdb_id}: {', '.join(keywords)}")     # 表示
    for kw in keywords:
        keyword_counter[kw] += 1  

# 上位キーワードの表示
top_keywords = keyword_counter.most_common(10)

# ヒストグラム表示
def save_keywords_histogram(counter, output_file="keywords_histogram.png"):
    labels, counts = zip(*counter)
    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"ヒストグラムを {output_file} に保存しました")

save_keywords_histogram(top_keywords, output_file="struct_keywords_histogram.png")