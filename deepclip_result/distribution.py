import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance

BANG_CSV   = "/home/slab/ishiiayuka/M2/deepclip_result/BAnG_RNAcompete.csv"
RANDOM_CSV = "/home/slab/ishiiayuka/M2/deepclip_result/Random_RNAcompete.csv"
aptamer_CSV = "/home/slab/ishiiayuka/M2/deepclip_result/aptamer_with_energy.csv"


CSV = "/home/slab/ishiiayuka/M2/deepclip_result/All_EFE_RNAcompete.csv"        
#COL = "GC_Content" 
#COL = "Length"      
COL = "EFE_Norm"

OUT_PNG = f"All_EFE_{COL}.png"

# KDEの滑らかさ（大きいほど “なだらか”）
BW = 1.2
# ========================


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
    aptamer = load_col(aptamer_CSV,COL)
    target = load_col(CSV, COL)  

    series = {
        "BAnG": bang,
        "Mymodel": target,
        "Random": rand,
        "aptamer": aptamer
    }

    stats = {}
    for name, arr in series.items():
        stats[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "n": int(len(arr)),
        }

    target_name = "aptamer"
    distances = {}
    for name, arr in series.items():
        if name == target_name:
            continue
        distances[name] = float(wasserstein_distance(series[target_name], arr))

    with open("output.log", "w") as f:
        f.write(f"Column: {COL}\n")
        f.write("Summary statistics (mean, std, n):\n")
        for name in series.keys():
            s = stats[name]
            f.write(f"  {name}: mean={s['mean']:.6f}, std={s['std']:.6f}, n={s['n']}\n")
        f.write("Wasserstein distances (vs aptamer):\n")
        for name in distances.keys():
            f.write(f"  {name}: {distances[name]:.6f}\n")

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
