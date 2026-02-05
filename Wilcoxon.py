import pandas as pd
from scipy.stats import wilcoxon

CSV_PATH = "Wilcoxon.csv"

BASELINES = ["RNA-BAnG", "Random"]
METHODS = ["method 1", "method 2", "method 3", "method 4", "method 5", "method 6"]

def read_csv_robust(path: str) -> pd.DataFrame:
    # Windows/Japanese環境も想定して複数エンコーディングを試す
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")

def wilcoxon_test_paired(df: pd.DataFrame, a_col: str, b_col: str):
    """
    a_col と b_col の対応あり Wilcoxon signed-rank（両側）
    - NaN行は除外
    - 0差は除外（zero_method="wilcox"）
    """
    pair = df[[a_col, b_col]].copy()
    pair[a_col] = pd.to_numeric(pair[a_col], errors="coerce")
    pair[b_col] = pd.to_numeric(pair[b_col], errors="coerce")
    pair = pair.dropna()

    a = pair[a_col].to_numpy()
    b = pair[b_col].to_numpy()
    d = a - b

    n_pairs = len(d)
    n_nonzero = int((d != 0).sum())

    if n_pairs == 0:
        return n_pairs, n_nonzero, None, None, "no valid pairs (all NaN)"
    if n_nonzero == 0:
        return n_pairs, n_nonzero, None, None, "all differences are zero (cannot test)"

    # SciPyのバージョン差を吸収（method引数が無い場合がある）
    try:
        stat, p = wilcoxon(
            a, b,
            zero_method="wilcox",
            alternative="two-sided",
            correction=True,
            method="auto",
        )
    except TypeError:
        stat, p = wilcoxon(
            a, b,
            zero_method="wilcox",
            alternative="two-sided",
            correction=True,
        )
    except ValueError as e:
        return n_pairs, n_nonzero, None, None, f"error: {e}"

    return n_pairs, n_nonzero, float(stat), float(p), ""

def main():
    df = read_csv_robust(CSV_PATH)

    # 列の存在チェック
    needed = set(BASELINES + METHODS)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSVに必要な列がありません: {missing}\n現在の列: {list(df.columns)}")

    for base in BASELINES:
        print(f"\n=== Wilcoxon signed-rank (two-sided):  {base}  vs  methods ===")
        print("comparison\t\t\tn_pairs\t n_nonzero\tstat\t\tp_value\tnote")
        for m in METHODS:
            n_pairs, n_nonzero, stat, p, note = wilcoxon_test_paired(df, m, base)
            stat_str = "NA" if stat is None else f"{stat:.6g}"
            p_str = "NA" if p is None else f"{p:.6g}"
            print(f"{m} vs {base}\t\t{n_pairs:>6}\t{n_nonzero:>9}\t{stat_str:>10}\t{p_str:>10}\t{note}")

if __name__ == "__main__":
    main()
