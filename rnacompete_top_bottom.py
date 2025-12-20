#!/usr/bin/env python3
"""
Download RNAcompete data and export the top/bottom N sequences by binding score.

Assumptions
- Each input table has a column for the RNA sequence and a binding score *or* a rank.
- If a rank column is used, lower rank = stronger binding (rank 1 is best).

Example
    python rnacompete_top_bottom.py \\
        --download-url https://example.org/RNAcompete_eukaryote.zip \\
        --extract-dir deepclip/data \\
        --output deepclip/rnacompete_top_bottom.csv
"""

import argparse
import csv
import gzip
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Set

try:
    import requests
except ImportError as e:  # pragma: no cover - requests is expected to be installed in this repo
    raise SystemExit("requests が見つかりません。pip install requests を実行してください。") from e


SequencePick = Tuple[str, float, str]  # sequence, metric_value, metric_col


def open_text_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r", newline="")


def download(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = dest_dir / Path(url).name
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(filename, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return filename


def extract(archive: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
        return extract_dir
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(extract_dir)
        return extract_dir
    # If it is already a directory, just return it.
    if archive.is_dir():
        return archive
    raise ValueError(f"Unknown archive format: {archive}")


def find_tables(base_dir: Path) -> Iterable[Path]:
    patterns = ("*.csv", "*.tsv", "*.txt")
    for pattern in patterns:
        yield from base_dir.rglob(pattern)


def pick_columns(header: List[str]) -> Tuple[str, str, bool]:
    """
    Returns (sequence_col, metric_col, lower_is_better)
    lower_is_better=True means smaller values are treated as better (e.g., rank).
    """
    header_lower = [h.lower() for h in header]
    seq_candidates = ("sequence", "seq", "rna_sequence", "rna", "probe")
    metric_candidates = (
        ("score", False),
        ("binding", False),
        ("intensity", False),
        ("enrichment", False),
        ("affinity", False),
        ("value", False),
    )
    rank_candidates = ("rank", "ranking", "order")

    seq_col = next((h for h in header if h.lower() in seq_candidates), None)
    if not seq_col:
        raise ValueError("No sequence column found in header: " + ",".join(header))

    for col, lower_is_better in metric_candidates:
        if col in header_lower:
            idx = header_lower.index(col)
            return seq_col, header[idx], lower_is_better

    for col in rank_candidates:
        if col in header_lower:
            idx = header_lower.index(col)
            return seq_col, header[idx], True

    raise ValueError("No metric column (score/rank) found in header: " + ",".join(header))


def parse_numeric(val: str) -> float:
    try:
        return float(val)
    except Exception:
        return float("nan")


def load_probe_to_rna(design_path: Path) -> dict:
    """
    RBD_v3_design_unified.txt:
    - tab-delimited
    - Probe ID in column 5 (0-based index 4)
    - RNA sequence near the end (DNA, RNA, structure, MFE)
    """
    if not design_path.exists():
        raise SystemExit(f"Design file not found: {design_path}")
    probe_to_rna = {}
    with open(design_path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, None)
        if not header:
            return probe_to_rna
        for row in reader:
            if len(row) < 5:
                continue
            probe_id = (row[4] or "").strip()
            if not probe_id:
                continue
            if len(row) >= 4:
                rna_seq = (row[-3] or "").strip()
            else:
                rna_seq = ""
            if rna_seq:
                probe_to_rna[probe_id] = rna_seq
    return probe_to_rna


def extract_from_norm_data(
    norm_path: Path,
    probe_to_rna: dict,
    allowed_proteins: Set[str],
    top_n: int,
    bottom_n: int,
) -> List[Tuple[str, str, str, int, str, float, str]]:
    if not norm_path.exists():
        raise SystemExit(f"Norm data file not found: {norm_path}")

    all_rows: List[Tuple[str, str, str, int, str, float, str]] = []
    with open_text_maybe_gzip(norm_path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, None)
        if not header:
            raise SystemExit(f"Empty norm data file: {norm_path}")

        # Identify base columns
        try:
            idx_rna = header.index("RNA_Seq")
        except ValueError:
            raise SystemExit("RNA_Seq column not found in norm data header")
        try:
            idx_probe = header.index("Probe_ID")
        except ValueError:
            idx_probe = None

        # Protein columns are the RNCMPT IDs
        protein_cols = []
        for i, col in enumerate(header):
            if col in allowed_proteins or col.split("_", 1)[0] in allowed_proteins:
                protein_cols.append((col, i))

        if not protein_cols:
            raise SystemExit("No RNCMPT columns matched the allowed protein list")

        # Collect rows per protein
        per_protein = {col: [] for col, _ in protein_cols}
        for row in reader:
            if len(row) <= idx_rna:
                continue
            rna_seq = (row[idx_rna] or "").strip()
            if not rna_seq and idx_probe is not None and len(row) > idx_probe:
                probe_id = (row[idx_probe] or "").strip()
                rna_seq = probe_to_rna.get(probe_id, "")
            if not rna_seq:
                continue

            for col, idx in protein_cols:
                if len(row) <= idx:
                    continue
                val = parse_numeric(row[idx])
                if val != val:
                    continue
                per_protein[col].append((rna_seq, val, col))

        for protein_id, records in per_protein.items():
            if not records:
                continue
            top_rows, bottom_rows = pick_top_bottom(records, False, top_n, bottom_n)
            for order, (seq, metric_val, col_name) in enumerate(top_rows, start=1):
                all_rows.append(
                    (protein_id, norm_path.name, "top", order, seq, metric_val, col_name)
                )
            for order, (seq, metric_val, col_name) in enumerate(bottom_rows, start=1):
                all_rows.append(
                    (protein_id, norm_path.name, "bottom", order, seq, metric_val, col_name)
                )

            print(f"{protein_id}: kept {len(top_rows)} top + {len(bottom_rows)} bottom")

    return all_rows


def load_allowed_proteins(csv_path: Path) -> Set[str]:
    if not csv_path.exists():
        raise SystemExit(f"Protein list CSV not found: {csv_path}")
    allowed: Set[str] = set()
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            protein_name = (row.get("protein_name") or "").strip()
            if file_name:
                stem = Path(file_name).stem
                allowed.add(stem)
                allowed.add(stem.split("_", 1)[0])
            if protein_name:
                allowed.add(protein_name)
    if not allowed:
        raise SystemExit(f"No proteins found in {csv_path}")
    return allowed


def load_rows(path: Path) -> Tuple[str, str, bool, List[SequencePick]]:
    with open(path, newline="") as fh:
        # Auto detect delimiter: tsv vs csv
        sample = fh.read(2048)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(fh, dialect=dialect)
        seq_col, metric_col, lower_is_better = pick_columns(reader.fieldnames or [])

        protein_id = path.stem
        records: List[SequencePick] = []
        for row in reader:
            seq = (row.get(seq_col) or "").strip()
            if not seq:
                continue
            metric_raw = row.get(metric_col)
            metric_val = parse_numeric(metric_raw) if metric_raw is not None else float("nan")
            if metric_val != metric_val:  # NaN check
                continue
            records.append((seq, metric_val, metric_col))
        return protein_id, metric_col, lower_is_better, records


def pick_top_bottom(records: List[SequencePick], lower_is_better: bool, top_n: int, bottom_n: int) -> Tuple[List[SequencePick], List[SequencePick]]:
    if not records:
        return [], []
    records_sorted = sorted(records, key=lambda r: r[1], reverse=not lower_is_better)
    top_part = records_sorted[:top_n]
    bottom_part = records_sorted[-bottom_n:] if bottom_n > 0 else []
    return top_part, bottom_part


def write_output(
    rows: List[Tuple[str, str, str, int, str, float, str]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["protein_id", "source_file", "bucket", "order", "sequence", "metric_value", "metric_column"]
        )
        for protein_id, source_file, bucket, order, seq, metric_val, metric_col in rows:
            writer.writerow([protein_id, source_file, bucket, order, seq, metric_val, metric_col])


def main():
    parser = argparse.ArgumentParser(description="RNAcompete top/bottom sequence extractor")
    parser.add_argument(
        "--download-url",
        help="RNAcompete zip/tar URL. If omitted, uses --extract-dir as already-downloaded data.",
    )
    parser.add_argument(
        "--extract-dir",
        default="deepclip/data",
        help="Directory to store/extract RNAcompete tables.",
    )
    parser.add_argument(
        "--design-file",
        default="RBD_v3_design_unified.txt",
        help="RBD_v3 design file (tab-delimited)",
    )
    parser.add_argument(
        "--norm-data",
        default="norm_data.txt.gz",
        help="RNAcompete normalized data file (tab-delimited, optionally gzipped)",
    )
    parser.add_argument("--output", default="deepclip/rnacompete_top_bottom.csv")
    parser.add_argument(
        "--protein-list-csv",
        default="RNAcompete.csv",
        help="Target proteins list (55 entries) from RNAcompete.csv",
    )
    parser.add_argument("--top", type=int, default=100, help="Top N sequences to keep")
    parser.add_argument("--bottom", type=int, default=100, help="Bottom N sequences to keep")
    args = parser.parse_args()

    extract_dir = Path(args.extract_dir)
    allowed_proteins = load_allowed_proteins(Path(args.protein_list_csv))
    archive_path: Optional[Path] = None

    # If norm_data is present, use the RBD/norm_data pipeline
    norm_path = Path(args.norm_data)
    design_path = Path(args.design_file)
    if norm_path.exists():
        probe_to_rna = load_probe_to_rna(design_path)
        all_rows = extract_from_norm_data(
            norm_path, probe_to_rna, allowed_proteins, args.top, args.bottom
        )
        write_output(all_rows, Path(args.output))
        print(f"Wrote {len(all_rows)} rows to {args.output}")
        return

    if args.download_url:
        print(f"Downloading {args.download_url} ...")
        archive_path = download(args.download_url, extract_dir)
        print(f"Saved archive to {archive_path}")
    else:
        print("No --download-url provided. Using existing files in extract-dir.")

    base_dir = extract_dir
    if archive_path:
        base_dir = extract(archive_path, extract_dir)
        print(f"Extracted to {base_dir}")
    elif not base_dir.exists():
        raise SystemExit(f"{base_dir} does not exist. Specify --download-url or create the directory with CSV/TSV files.")

    tables = list(find_tables(base_dir))
    if not tables:
        raise SystemExit(f"No CSV/TSV files found under {base_dir}")

    all_rows: List[Tuple[str, str, str, int, str, float, str]] = []  # protein_id, source, bucket, order, seq, metric, metric_col
    for table in tables:
        try:
            protein_id, metric_col, lower_is_better, records = load_rows(table)
        except Exception as e:
            print(f"[WARN] Skipping {table}: {e}")
            continue
        if (
            protein_id not in allowed_proteins
            and protein_id.split("_", 1)[0] not in allowed_proteins
        ):
            continue

        top_rows, bottom_rows = pick_top_bottom(records, lower_is_better, args.top, args.bottom)

        # Build output rows with per-protein order
        for order, (seq, metric_val, col_name) in enumerate(top_rows, start=1):
            all_rows.append((protein_id, table.name, "top", order, seq, metric_val, col_name))
        for order, (seq, metric_val, col_name) in enumerate(bottom_rows, start=1):
            all_rows.append((protein_id, table.name, "bottom", order, seq, metric_val, col_name))

        print(f"{table.name}: kept {len(top_rows)} top + {len(bottom_rows)} bottom")

    write_output(all_rows, Path(args.output))
    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
