#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def split_by_bucket(input_path: Path, out_top: Path, out_bottom: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        required = {"id", "bucket", "sequence"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        top_rows = []
        bottom_rows = []
        for row in reader:
            bucket = (row.get("bucket") or "").strip()
            if bucket == "top":
                top_rows.append(row)
            elif bucket == "bottom":
                bottom_rows.append(row)

    if top_rows:
        with out_top.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(top_rows)

    if bottom_rows:
        with out_bottom.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(bottom_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split CSV by bucket column into top/bottom files"
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=Path("rnacompete_top_bottom.csv"),
        type=Path,
        help="Input CSV path (default: rnacompete_top_bottom.csv)",
    )
    parser.add_argument(
        "--out-top",
        type=Path,
        default=Path("top.csv"),
        help="Output CSV for bucket=top",
    )
    parser.add_argument(
        "--out-bottom",
        type=Path,
        default=Path("bottom.csv"),
        help="Output CSV for bucket=bottom",
    )
    args = parser.parse_args()
    split_by_bucket(args.input_csv, args.out_top, args.out_bottom)


if __name__ == "__main__":
    main()
