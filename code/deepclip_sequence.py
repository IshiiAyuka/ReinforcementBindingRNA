import requests
import pandas as pd
from tqdm import tqdm
import time

# === 入力ファイルパス ===
pretrain_csv_path = "pretrained_models.csv"
output_csv_path = "uniprot_sequences.csv"

# === タンパク質名リストを取得（重複除去）===
pretrain_df = pd.read_csv(pretrain_csv_path, low_memory=False)
protein_names = pretrain_df["protein"].dropna().unique()

# === UniProt APIから配列を取得する関数 ===
def fetch_uniprot_sequences(gene_name):
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"gene:{gene_name} AND organism_id:9606 AND reviewed:true",
        "fields": "accession,sequence",
        "format": "tsv",
        "size": 10
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            lines = response.text.strip().split("\n")[1:]  # ヘッダー除去
            if not lines:
                print(f"  → UniProtヒットなし: {gene_name}", flush=True)
                return []
            sequences = []
            for line in lines:
                acc, seq = line.split("\t")
                sequences.append((gene_name, acc, seq))
            print(f"  → UniProtヒット {len(sequences)} 件: {gene_name}", flush=True)
            return sequences
        else:
            print(f"  → API失敗: {gene_name} (status code: {response.status_code})", flush=True)
            return []
    except Exception as e:
        print(f"  → リクエストエラー: {gene_name} - {e}", flush=True)
        return []

# === 取得したすべての結果を格納するリスト ===
all_results = []

# === メイン処理 ===
for name in tqdm(protein_names, desc="UniProt検索中"):
    entries = fetch_uniprot_sequences(name)
    all_results.extend(entries)
    time.sleep(1)  # API制限対策

# === DataFrameに変換してCSV保存 ===
if all_results:
    result_df = pd.DataFrame(all_results, columns=["protein_name", "uniprot_id", "sequence"])
    result_df.to_csv(output_csv_path, index=False)
    print(f"\n出力完了: {output_csv_path}", flush=True)
else:
    print("\n 取得できた配列がありませんでした。", flush=True)
