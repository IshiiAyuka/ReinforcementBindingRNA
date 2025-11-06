from tqdm import tqdm
import pandas as pd
import requests
from collections import Counter
import matplotlib.pyplot as plt
import time

# CSV から ID 列を読み込み
df = pd.read_csv("uniprot_sequences.csv")
ids = df["uniprot_id"].dropna().unique().tolist()

keywords_list = []
headers = {"Accept": "application/json"}

# tqdm で進捗表示
for uid in tqdm(ids, desc="Fetching UniProt keywords"):
    url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"Failed to fetch {uid}: {resp.status_code} {resp.reason}")
        continue

    entry = resp.json()
    # "keywords" 配列の中から category が "Molecular function" の name のみを抽出
    matched = []
    for kw in entry.get("keywords", []):
        # Uniprot JSON では kw["category"] にカテゴリ名が入っている
        if kw.get("category") == "Molecular function":
            name = kw.get("name", "")
            matched.append(name)
            keywords_list.append(name)

    # IDごとに何が取れたかを出力
    if matched:
        print(f"{uid} → Molecular function: {', '.join(matched)}")
    else:
        print(f"{uid} → (no molecular functions)")

    time.sleep(0.1)

# ここからは集計・グラフ描画（前と同じ）
kw_counts = Counter(keywords_list)
kw_df = pd.DataFrame.from_records(
    list(kw_counts.items()),
    columns=["keyword", "count"]
).sort_values("count", ascending=False)

plt.figure(figsize=(8, 6))
plt.bar(kw_df["keyword"], kw_df["count"])
plt.xticks(rotation=45, ha="right")
plt.xlabel("Molecular function")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("molecular_function_counts.png", dpi=300)
plt.show()
