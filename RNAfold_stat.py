#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
import subprocess
import sys
from typing import Iterable, List, Tuple


def gc_content(sequence: str) -> float:
    seq = sequence.upper()
    if not seq:
        return 0.0
    gc_count = sum(1 for base in seq if base in ("G", "C"))
    return gc_count / len(seq)


def run_rnafold(sequence: str) -> Tuple[float, float]:
    """Run RNAfold for a single sequence and return (mfe, ensemble_energy)."""
    proc = subprocess.run(
        ["RNAfold", "--noPS", "-p"],
        input=sequence + "\n",
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"RNAfold failed: {proc.stderr.strip()}")

    mfe = None
    ensemble_energy = None
    energies: List[float] = []

    for line in proc.stdout.splitlines():
        energy_match = re.search(r"[\(\[\{]\s*([-+]?\d+(?:\.\d+)?)", line)
        if energy_match:
            energy_val = float(energy_match.group(1))
            energies.append(energy_val)
            if mfe is None and "(" in line:
                mfe = energy_val

        if "free energy of ensemble" in line:
            ensemble_match = re.search(r"ensemble\s*=\s*([-+]?\d+(?:\.\d+)?)", line)
            if ensemble_match:
                ensemble_energy = float(ensemble_match.group(1))

    if ensemble_energy is None and energies:
        # Some RNAfold builds omit the explicit "free energy of ensemble" line.
        # Use the last reported energy (often in [] or {} for centroid/MEA) as a fallback.
        ensemble_energy = energies[-1]

    if mfe is None or ensemble_energy is None:
        raise ValueError(
            "Could not parse energies from RNAfold output:\n"
            f"{proc.stdout}"
        )
    return mfe, ensemble_energy


def load_rows(csv_path: str) -> Tuple[List[dict], List[str]]:
    rows: List[dict] = []
    sequences: List[str] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "sequence" not in reader.fieldnames:
            raise KeyError("Sequence column not found in CSV header")
        for row in reader:
            seq = (row.get("sequence") or "").strip()
            rows.append(row)
            if seq:
                sequences.append(seq)
    return rows, sequences


def summarize(values: Iterable[float]) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        raise ValueError("No values provided for summary")
    mean_val = statistics.mean(vals)
    std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean_val, std_val


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute MFE and ensemble free energy statistics from RNAfold"
    )
    parser.add_argument(
        "--csv",
        default="/home/slab/ishiiayuka/M2/deepclip/generated_rna/generated_rna_1212_All_EFE.csv",
        help="Path to input CSV containing a Sequence column (default: aptamer.csv)",
    )
    parser.add_argument(
        "--out",
        default="All_EFE_RNAcompete.csv",
        help="Path to write CSV with added MFE and EnsembleEnergy columns (default: aptamer_with_energy.csv)",
    )
    args = parser.parse_args()

    try:
        rows, sequences = load_rows(args.csv)
    except Exception as exc:
        sys.stderr.write(f"Failed to load sequences: {exc}\n")
        return 1

    if not sequences:
        sys.stderr.write("No sequences found to process.\n")
        return 1

    mfe_values: List[float] = []
    ensemble_values: List[float] = []
    failures: List[Tuple[int, str]] = []

    for idx, (seq, row) in enumerate(zip(sequences, rows), start=1):
        truncated_seq = seq[:100] if len(seq) >= 101 else seq
        seq_len = len(truncated_seq)
        gc_frac = gc_content(truncated_seq)
        row["Length"] = str(seq_len)
        row["GC_Content"] = f"{gc_frac:.4f}"
        try:
            mfe, ensemble = run_rnafold(truncated_seq)
            mfe_values.append(mfe)
            ensemble_values.append(ensemble)
            row["MFE"] = f"{mfe:.4f}"
            row["EnsembleEnergy"] = f"{ensemble:.4f}"
            if seq_len:
                row["MFE_per_len"] = f"{(mfe / seq_len):.4f}"
                row["EnsembleEnergy_per_len"] = f"{(ensemble / seq_len):.4f}"
            else:
                row["MFE_per_len"] = ""
                row["EnsembleEnergy_per_len"] = ""
        except Exception as exc:
            failures.append((idx, str(exc)))
            row["MFE"] = ""
            row["EnsembleEnergy"] = ""
            row["MFE_per_len"] = ""
            row["EnsembleEnergy_per_len"] = ""

    if args.out:
        fieldnames = list(rows[0].keys())
        if "MFE" not in fieldnames:
            fieldnames += ["MFE", "EnsembleEnergy"]
        if "Length" not in fieldnames:
            fieldnames += ["Length", "GC_Content"]
        if "MFE_per_len" not in fieldnames:
            fieldnames += ["MFE_per_len", "EnsembleEnergy_per_len"]
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if failures:
        sys.stderr.write(f"{len(failures)} sequences failed to fold:\n")
        for idx, msg in failures:
            sys.stderr.write(f"  #{idx}: {msg}\n")

    if not mfe_values or not ensemble_values:
        sys.stderr.write("No energies were computed.\n")
        return 1

    mfe_mean, mfe_std = summarize(mfe_values)
    ensemble_mean, ensemble_std = summarize(ensemble_values)

    print(f"Sequences processed: {len(sequences)}")
    print(f"MFE mean: {mfe_mean:.4f} kcal/mol")
    print(f"MFE std:  {mfe_std:.4f} kcal/mol")
    print(f"Ensemble free energy mean: {ensemble_mean:.4f} kcal/mol")
    print(f"Ensemble free energy std:  {ensemble_std:.4f} kcal/mol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
