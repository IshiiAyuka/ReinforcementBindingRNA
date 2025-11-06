# -*- coding: utf-8 -*-

import os
import pandas as pd

# 入力ファイル名（必要に応じて変更してください）
input_csv = "generated_rna_trimmed_12_75.csv"

# 出力先ディレクトリ
output_dir = "data"

# 出力ディレクトリがなければ作成
os.makedirs(output_dir, exist_ok=True)

# CSV を読み込む
df = pd.read_csv(input_csv)

# id 列でグループ化して、各グループを個別ファイルに書き出し
for uid, group in df.groupby("id"):
    output_path = os.path.join(output_dir, f"{uid}.csv")
    # index=False で行番号を出力しない、ヘッダーはデフォルトで出力される
    group.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
