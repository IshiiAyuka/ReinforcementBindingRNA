import pandas as pd
import re

# CSVファイルのパス
csv_path = "ppi3d.csv"  # ここをファイル名に置き換えてください

# CSVファイルを読み込む
df = pd.read_csv(csv_path)

'''# s2_sequenceカラムにAUGC以外の文字が含まれる行を除外
pattern = re.compile(r'^[AUGC]+$')
filtered_df = df[df["s2_sequence"].apply(lambda x: bool(pattern.match(str(x).upper())))]'''

# s1_number_of_residues が数値で 1025 未満の行だけを抽出
mask_len = pd.to_numeric(df["s1_number_of_residues"], errors="coerce") < 1025
filtered_df = df[mask_len]

# 結果を新しいCSVに保存（必要に応じて）
filtered_df.to_csv("filtered_ppi3d.csv", index=False)

print(f"削除前の行数: {len(df)}")
print(f"削除後の行数: {len(filtered_df)}")
