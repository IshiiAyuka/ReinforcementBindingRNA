import pandas as pd

# CSV 読み込み（必要に応じて sep='\t' 等を指定）
df = pd.read_csv("ppi3d.csv", low_memory=False)
df.columns = df.columns.str.strip().str.lower()

# 欠損値がある場合に備えて文字列化
seqs = df["s2_sequence"].astype(str)
lengths = seqs.str.len()

# 各配列の長さを取得し、12～75 の範囲にあるものを真とするブールマスク
mask = lengths.between(12, 1000)

# カウント
count = mask.sum()

print(f"長さ12〜75塩基の配列は {count} 個あります。")
