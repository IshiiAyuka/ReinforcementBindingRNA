#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gzip
import urllib.request
import urllib.error
from typing import Optional, Tuple
from multiprocessing import Pool

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SciPy があれば高速に近傍探索
try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def _clean_len(val) -> Optional[int]:
    """空白除去・大小無視で長さを返す。欠損は None。"""
    if pd.isna(val):
        return None
    s = "".join(str(val).split())
    return len(s) if s != "" else 0


def atoms_of_chain(chain) -> np.ndarray:
    """標準残基のみ（res.id[0]==' '）、H除外、altlocは' 'または'A'。返り値: (N,3) float32"""
    coords = []
    for res in chain.get_residues():
        if res.id[0] != ' ':
            continue
        for atom in res.get_atoms():
            if (atom.element or '').upper() == 'H':
                continue
            altloc = atom.get_altloc()
            if altloc not in (' ', 'A'):
                continue
            coords.append(atom.coord)
    return np.asarray(coords, dtype=np.float32) if coords else np.empty((0, 3), dtype=np.float32)


# ---------- PDB からの自動ダウンロード ----------
def _fetch_bytes(url: str, timeout: int = 30) -> Optional[bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError):
        return None


def ensure_cif_gz(pdb_id_lower: str, data_dir: str) -> Optional[str]:
    """
    {data_dir}/{pdb_id}.cif.gz が無ければ RCSB からダウンロードして用意する。
    まず .cif.gz を試し、無ければ .cif を取得して gzip 化。
    成功時は .cif.gz のパス、失敗時は None。
    """
    os.makedirs(data_dir, exist_ok=True)
    pdb_uc = pdb_id_lower.upper()
    path_gz = os.path.join(data_dir, f"{pdb_id_lower}.cif.gz")

    if os.path.exists(path_gz):
        return path_gz

    # 1) .cif.gz
    url_gz = f"https://files.rcsb.org/download/{pdb_uc}.cif.gz"
    data = _fetch_bytes(url_gz)
    if data:
        try:
            with open(path_gz, "wb") as f:
                f.write(data)
            return path_gz
        except Exception:
            pass

    # 2) .cif を取得して gzip 化
    url_cif = f"https://files.rcsb.org/download/{pdb_uc}.cif"
    data = _fetch_bytes(url_cif)
    if data:
        try:
            with gzip.open(path_gz, "wb") as gz:
                gz.write(data)
            return path_gz
        except Exception:
            pass

    return None
# ----------------------------------------------------


def process_row(row, data_dir: str, cutoff: float) -> Tuple[Optional[str], str]:
    """
    1行処理。
    戻り値:
      (出力メッセージ or None, ステータス 'ok'|'err'|'skip')
    'skip' の場合は print しない。
    """
    pdb_id = str(row["pdb_id"]).lower()
    protein_chain = str(row["subunit_1"]).strip().split("_")[-1]
    rna_chain     = str(row["subunit_2"]).strip().split("_")[-1]

    # --- 長さフィルタ（CSV: s1_sequence / s2_sequence から取得） ---
    prot_len = _clean_len(row.get("s1_sequence"))
    rna_len  = _clean_len(row.get("s2_sequence"))

    # 欠損はスキップ（無言）
    if prot_len is None or rna_len is None:
        return (None, "skip")

    # --- mmCIF を用意（無ければダウンロード） ---
    cif_file = os.path.join(data_dir, f"{pdb_id}.cif.gz")
    if not os.path.exists(cif_file):
        fetched = ensure_cif_gz(pdb_id, data_dir)
        if fetched is None:
            msg = (f"[{pdb_id.upper()}] {protein_chain}-{rna_chain} "
                   f"rna_res=NA rna_res_contact=NA rna_contact_ratio=NA "
                   f"note=missing_cif_and_download_failed")
            return (msg, "err")
        cif_file = fetched

    # --- ここから計算 ---
    parser = MMCIFParser(QUIET=True)
    with gzip.open(cif_file, "rt") as fh:
        structure = parser.get_structure("complex", fh)
    model = next(structure.get_models(), None)
    if model is None:
        msg = (f"[{pdb_id.upper()}] {protein_chain}-{rna_chain} "
               f"rna_res=NA rna_res_contact=NA rna_contact_ratio=NA note=no_model")
        return (msg, "err")

    try:
        chain_p = model[protein_chain]
        chain_r = model[rna_chain]
    except KeyError:
        msg = (f"[{pdb_id.upper()}] {protein_chain}-{rna_chain} "
               f"rna_res=NA rna_res_contact=NA rna_contact_ratio=NA note=missing_chain")
        return (msg, "err")

    # RNA残基数（標準残基のみ）
    r_res = sum(1 for res in chain_r.get_residues() if res.id[0] == ' ')
    # 接触しているRNA残基数を計算
    rna_res_contact = 0
    if r_res > 0:
        coords_p = atoms_of_chain(chain_p)
        coords_r = atoms_of_chain(chain_r)
        if coords_p.size and coords_r.size:
            # RNA各原子がどの残基に属するか
            rna_atom2res = []
            ridx = -1
            for res in chain_r.get_residues():
                if res.id[0] != ' ':
                    continue
                ridx += 1
                for atom in res.get_atoms():
                    if (atom.element or '').upper() == 'H':
                        continue
                    altloc = atom.get_altloc()
                    if altloc in (' ', 'A'):
                        rna_atom2res.append(ridx)
            rna_atom2res = np.asarray(rna_atom2res, dtype=np.int32)

            if rna_atom2res.size:
                if HAVE_SCIPY:
                    tree_r = cKDTree(coords_r)
                    neigh = tree_r.query_ball_point(coords_p, r=cutoff)
                    contact_atoms = []
                    for lst in neigh:
                        if lst:
                            contact_atoms.extend(lst)
                    if contact_atoms:
                        rna_res_contact = int(np.unique(rna_atom2res[np.unique(contact_atoms)]).size)
                else:
                    r2 = cutoff * cutoff
                    BLOCK = 4096
                    rna_atom_contact_mask = np.zeros(coords_r.shape[0], dtype=bool)
                    for i in range(0, len(coords_p), BLOCK):
                        A = coords_p[i:i + BLOCK]
                        diff = A[:, None, :] - coords_r[None, :, :]
                        d2 = np.einsum('ijk,ijk->ij', diff, diff, optimize=True)
                        rna_atom_contact_mask |= (d2 <= r2).any(axis=0)
                    rna_res_contact = int(np.unique(rna_atom2res[rna_atom_contact_mask]).size)

    ratio_str = f"{(rna_res_contact / r_res):.3f}" if r_res > 0 else "NA"
    msg = (f"[{pdb_id.upper()}] {protein_chain}-{rna_chain} "
           f"rna_res={r_res} rna_res_contact={rna_res_contact} "
           f"rna_contact_ratio={ratio_str}")
    return (msg, "ok")


def process_row_safe(row, data_dir: str, cutoff: float) -> Tuple[Optional[str], str]:
    try:
        return process_row(row, data_dir, cutoff)
    except Exception as e:
        pdb_id = str(row.get("pdb_id", "NA")).upper()
        msg = f"[{pdb_id}] rna_res=NA rna_res_contact=NA rna_contact_ratio=NA note=exception"
        return (msg, "err")


def process_row_safe_star(args) -> Tuple[Optional[str], str]:
    row, data_dir, cutoff = args
    return process_row_safe(row, data_dir, cutoff)


def main():
    # ここで全ての設定を指定
    csv_path = "/home/slab/ishiiayuka/M2/ppi3d.csv"
    data_dir = "./home/slab/ishiiayuka/M2/mmcif_protein_rna"
    cutoff = 5.0
    no_error_logs = False
    hist_png = "contact.png"
    hist_bins = 10
    use_multiprocessing = True
    num_workers = max(1, (os.cpu_count() or 2) - 1)
    chunk_size = 20

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    # 欠損数の確認
    total_rows = len(df)
    missing_s1 = df["s1_sequence"].isna().sum()
    missing_s2 = df["s2_sequence"].isna().sum()
    tqdm.write(f"total_rows={total_rows} missing_s1_sequence={missing_s1} missing_s2_sequence={missing_s2}")

    # 欠損のみ除外（長さフィルタなし）
    prot_len = df["s1_sequence"].apply(_clean_len)
    rna_len = df["s2_sequence"].apply(_clean_len)
    mask = prot_len.notna() & rna_len.notna()
    df = df.loc[mask, ["pdb_id", "subunit_1", "subunit_2", "s1_sequence", "s2_sequence"]]
    records = df.to_dict("records")
    tqdm.write(f"records_after_filter={len(records)}")

    ok = err = skipped = 0
    ratios = []
    if use_multiprocessing and num_workers > 1:
        tasks = ((row, data_dir, cutoff) for row in records)
        with Pool(processes=num_workers, maxtasksperchild=200) as pool:
            it = pool.imap_unordered(
                process_row_safe_star,
                tasks,
                chunksize=chunk_size,
            )
            for msg, status in tqdm(it, total=len(records), desc="原子間コンタクト集計中", ncols=100):
                if status == "ok":
                    tqdm.write(msg)
                    try:
                        ratio_val = float(msg.split("rna_contact_ratio=")[-1])
                        ratios.append(ratio_val)
                    except Exception:
                        pass
                    ok += 1
                elif status == "err":
                    if not no_error_logs:
                        tqdm.write(msg)
                    err += 1
                else:  # skip
                    skipped += 1  # 無言
    else:
        for row in tqdm(records, total=len(records), desc="原子間コンタクト集計中", ncols=100):
            msg, status = process_row_safe(row, data_dir, cutoff)
            if status == "ok":
                tqdm.write(msg)
                try:
                    ratio_val = float(msg.split("rna_contact_ratio=")[-1])
                    ratios.append(ratio_val)
                except Exception:
                    pass
                ok += 1
            elif status == "err":
                if not no_error_logs:
                    tqdm.write(msg)
                err += 1
            else:  # skip
                skipped += 1  # 無言
    tqdm.write(f"done: ok={ok}, error={err}, skipped={skipped}")
    if hist_png and ratios:
        plt.figure(figsize=(6, 4))
        plt.hist(ratios, bins=hist_bins, color="#4C78A8", edgecolor="white")
        plt.xlabel("RNA contact ratio")
        plt.ylabel("Count")
        plt.title("RNA contact ratio histogram")
        plt.tight_layout()
        plt.savefig(hist_png, dpi=200)


if __name__ == "__main__":
    main()
