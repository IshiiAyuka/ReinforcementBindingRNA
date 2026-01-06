import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

BANG_CSV   = "/home/slab/ishiiayuka/M2/deepclip_result/BAnG_RNAcompete.csv"
RANDOM_CSV = "/home/slab/ishiiayuka/M2/deepclip_result/Random_RNAcompete.csv"
top_CSV = "/home/slab/ishiiayuka/M2/deepclip_result/top_RNAcompete.csv"
bottom_CSV = "/home/slab/ishiiayuka/M2/deepclip_result/bottom_RNAcompete.csv"

CSV = "/home/slab/ishiiayuka/M2/deepclip_result/LucaOneOnly_RNAcompete.csv"        
COL = "Length"       

OUT_PNG = f"LucaOneOnly_{COL}.png"

# KDEの滑らかさ（大きいほど “なだらか”）
BW = 1.2
# =========================


def load_col(path: str, col: str) -> np.ndarray:
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    if col not in df.columns:
        raise KeyError(f"{path}: column '{col}' が見つかりません。列名: {list(df.columns)}")

    arr = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
    if len(arr) < 3:
        raise ValueError(f"{path}: '{col}' の有効データが少なすぎます（{len(arr)}件）。")
    return arr


def main():

    bang = load_col(BANG_CSV, COL)
    rand = load_col(RANDOM_CSV, COL)
    top = load_col(top_CSV,COL)
    bottom = load_col(bottom_CSV,COL)
    target = load_col(CSV, COL)  

    series = {
        "BAnG": bang,
        "Mymodel": target,
        "Random": rand,
        "top": top,
        "bottom":bottom
    }

    all_vals = np.concatenate(list(series.values()))
    x = np.linspace(all_vals.min(), all_vals.max(), 400)

    plt.figure(figsize=(7, 4))
    for name, arr in series.items():
        kde = gaussian_kde(arr, bw_method=BW)
        y = kde(x)
        plt.plot(x, y, label=f"{name} (n={len(arr)})")
        plt.fill_between(x, y, alpha=0.25)

    plt.xlabel(COL)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"[saved] {OUT_PNG}")

    plt.show()


if __name__ == "__main__":
    main()
