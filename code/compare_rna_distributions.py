import csv
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt

FILES = [
    "deepclip/generated_rna/generated_rna_1212_DecoderOnly.csv",
    "deepclip/generated_rna/BAnG_result.csv",
    "deepclip/generated_rna/generated_rna_1212_Random.csv",
]

SEQ_COLUMN = "sequence"

OUTPUT_DIR = Path("rna_distribution_plots")
RNAFOLD_BIN = "RNAfold"


def gc_percent(seq: str) -> Optional[float]:
    seq = (seq or "").strip().upper()
    if not seq:
        return None
    gc = sum(1 for ch in seq if ch in ("G", "C"))
    return 100.0 * gc / len(seq)


def run_rnafold(seq_id: str, seq: str) -> Tuple[Optional[float], Optional[float]]:
    seq = (seq or "").strip().upper()
    if not seq:
        return None, None

    if shutil.which(RNAFOLD_BIN) is None:
        raise RuntimeError(f"{RNAFOLD_BIN} not found in PATH.")

    input_text = f">{seq_id}\n{seq}\n"
    result = subprocess.run(
        [RNAFOLD_BIN, "-p", "--noPS"],
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )

    mfe = None
    ens = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Structure line includes MFE in parentheses at the end: ".... (-7.80)"
        if "(" in line and ")" in line:
            match = re.search(r"\(([-\d\.]+)\)\s*$", line)
            if match:
                mfe = float(match.group(1))
        if "ensemble free energy" in line:
            match = re.search(r"([-\\d\\.]+)", line)
            if match:
                ens = float(match.group(1))
        if "free energy of ensemble" in line:
            match = re.search(r"([-\\d\\.]+)", line)
            if match:
                ens = float(match.group(1))

    return mfe, ens


def load_values(path: str):
    gc_vals = []
    len_vals = []
    mfe_vals = []
    ens_vals = []
    energy_rows = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row.get("id", "")
            seq = row.get(SEQ_COLUMN, "")
            gc = gc_percent(seq)
            if gc is not None:
                gc_vals.append(gc)
                len_vals.append(len(seq.strip()))

            mfe, ens = run_rnafold(seq_id, seq)
            if mfe is not None:
                mfe_vals.append(mfe)
            if ens is not None:
                ens_vals.append(ens)
            energy_rows.append(
                {
                    "id": seq_id,
                    "sequence": seq,
                    "mfe": mfe if mfe is not None else "",
                    "ensemble_free_energy": ens if ens is not None else "",
                }
            )

    if energy_rows:
        out_csv = OUTPUT_DIR / f"{Path(path).stem}_energies.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "sequence", "mfe", "ensemble_free_energy"]
            )
            writer.writeheader()
            writer.writerows(energy_rows)

    return {
        "gc": gc_vals,
        "length": len_vals,
        "mfe": mfe_vals,
        "ensemble": ens_vals,
    }


def plot_metric(metric_name: str, all_values: dict, xlabel: str, out_path: Path):
    plt.figure(figsize=(8, 5))
    bins = 40

    plotted = False
    for label, values in all_values.items():
        if not values:
            continue
        plt.hist(values, bins=bins, alpha=0.5, density=True, label=label)
        plotted = True

    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(f"{metric_name} distribution")
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_gc = {}
    all_len = {}
    all_mfe = {}
    all_ens = {}

    for path in FILES:
        label = Path(path).stem
        values = load_values(path)
        all_gc[label] = values["gc"]
        all_len[label] = values["length"]
        all_mfe[label] = values["mfe"]
        all_ens[label] = values["ensemble"]

    plot_metric("GC content", all_gc, "GC content (%)", OUTPUT_DIR / "gc_distribution.png")
    plot_metric("Sequence length", all_len, "Length (nt)", OUTPUT_DIR / "length_distribution.png")
    plot_metric("MFE", all_mfe, "MFE (kcal/mol)", OUTPUT_DIR / "mfe_distribution.png")
    plot_metric(
        "Ensemble free energy",
        all_ens,
        "Ensemble free energy (kcal/mol)",
        OUTPUT_DIR / "ensemble_free_energy_distribution.png",
    )


if __name__ == "__main__":
    main()
