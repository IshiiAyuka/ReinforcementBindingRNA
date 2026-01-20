# rle_diversity_compare_two_csvs.py
import math
from collections import Counter
import pandas as pd

# =========================
# 設定（ここだけ編集）
# =========================
CSV_A_PATH = "/home/slab/ishiiayuka/M2/deepclip_result/BAnG_RNAcompete.csv"   # モデルA
CSV_B_PATH = "/home/slab/ishiiayuka/M2/deepclip_result/All_MFE_RNAcompete_with_score.csv"  # モデルB（ここを差し替え）
LABEL_A = "A"
LABEL_B = "B"

ID_COL  = "id"
SEQ_COL = "sequence"

RUN_K = 4                 # 診断用（AAAAAなどの連続をどれくらいで“長い”とみなすか）
PRINT_MODE = "csv"      # "table" or "csv"
# =========================


def clean_seq(s: str) -> str:
    return str(s).strip().upper().replace("T", "U")


def collapse_runs(seq: str) -> str:
    if not seq:
        return ""
    out = [seq[0]]
    for ch in seq[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def rle(seq: str):
    if not seq:
        return [], []
    bases = [seq[0]]
    lens = [1]
    for ch in seq[1:]:
        if ch == bases[-1]:
            lens[-1] += 1
        else:
            bases.append(ch)
            lens.append(1)
    return bases, lens


def shannon_entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if n == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log(p + 1e-300)
    return h


def summarize_one_protein(seqs, run_k=4):
    n = len(seqs)
    raw_unique = len(set(seqs))

    collapsed = [collapse_runs(s) for s in seqs]
    type_counts = Counter(collapsed)

    neff_rle = float(len(type_counts))          # collapsed type の数
    neff_rle_norm = neff_rle / n if n else 0.0  # 0..1

    h_type = shannon_entropy(type_counts)

    max_runs = []
    excess_runs = []
    for s in seqs:
        _, lens = rle(s)
        lmax = max(lens) if lens else 0
        excess = sum(max(0, L - run_k) for L in lens)
        max_runs.append(lmax)
        excess_runs.append(excess)

    return {
        "N": n,
        "unique_raw": raw_unique,
        "unique_rle_type": int(neff_rle),
        "neff_rle_norm": neff_rle_norm,
        "type_entropy_rle": h_type,
        "mean_max_run": sum(max_runs) / n if n else 0.0,
        f"mean_excess_run_k{run_k}": sum(excess_runs) / n if n else 0.0,
    }


def compute_stats(csv_path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if ID_COL not in df.columns:
        raise SystemExit(f"[{label}] ID_COL='{ID_COL}' が見つかりません。columns={list(df.columns)}")
    if SEQ_COL not in df.columns:
        raise SystemExit(f"[{label}] SEQ_COL='{SEQ_COL}' が見つかりません。columns={list(df.columns)}")

    df[SEQ_COL] = df[SEQ_COL].map(clean_seq)

    rows = []
    for pid, g in df.groupby(ID_COL, sort=False):
        seqs = g[SEQ_COL].tolist()
        st = summarize_one_protein(seqs, run_k=RUN_K)
        st[ID_COL] = pid
        rows.append(st)

    out = pd.DataFrame(rows)

    # 列名にラベルを付ける（後で横結合するため）
    rename = {c: f"{c}_{label}" for c in out.columns if c != ID_COL}
    out = out.rename(columns=rename)
    return out


def main():
    a = compute_stats(CSV_A_PATH, LABEL_A)
    b = compute_stats(CSV_B_PATH, LABEL_B)

    # id で結合（片方にしかないidも出せるよう outer）
    m = a.merge(b, on=ID_COL, how="outer")

    # 比較したい主指標：neff_rle_norm（高いほど多様）
    key_a = f"neff_rle_norm_{LABEL_A}"
    key_b = f"neff_rle_norm_{LABEL_B}"

    m["delta_neff_rle_norm"] = m[key_a] - m[key_b]
    m["winner_neff_rle_norm"] = m["delta_neff_rle_norm"].apply(
        lambda x: LABEL_A if x > 0 else (LABEL_B if x < 0 else "tie")
    )

    # 集計
    total = m[ID_COL].notna().sum()
    win_a = (m["winner_neff_rle_norm"] == LABEL_A).sum()
    win_b = (m["winner_neff_rle_norm"] == LABEL_B).sum()
    tie   = (m["winner_neff_rle_norm"] == "tie").sum()

    # 表示用に列を整理
    cols = [
        ID_COL,
        f"N_{LABEL_A}", f"N_{LABEL_B}",
        f"unique_raw_{LABEL_A}", f"unique_raw_{LABEL_B}",
        f"unique_rle_type_{LABEL_A}", f"unique_rle_type_{LABEL_B}",
        key_a, key_b,
        "delta_neff_rle_norm", "winner_neff_rle_norm",
        f"mean_max_run_{LABEL_A}", f"mean_max_run_{LABEL_B}",
        f"mean_excess_run_k{RUN_K}_{LABEL_A}", f"mean_excess_run_k{RUN_K}_{LABEL_B}",
    ]
    cols = [c for c in cols if c in m.columns]
    m = m[cols].sort_values("delta_neff_rle_norm", ascending=False)

    # まずサマリをprint
    print("=== Diversity comparison (higher neff_rle_norm = more diverse) ===")
    print(f"metric: neff_rle_norm  |  {LABEL_A}: {CSV_A_PATH}  vs  {LABEL_B}: {CSV_B_PATH}")
    print(f"ids compared (union): {total}")
    print(f"wins: {LABEL_A}={win_a}, {LABEL_B}={win_b}, tie={tie}")
    print()

    # 詳細表
    if PRINT_MODE == "csv":
        print(m.to_csv(index=False), end="")
    else:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 220)
        print(m.to_string(index=False))


if __name__ == "__main__":
    main()
