#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from Bio.PDB import PDBParser, NeighborSearch, Polypeptide


# ----------------------------
# 文字→塩基 変換（PDB residue name）
# ----------------------------
BASE_MAP = {
    "A": "A", "C": "C", "G": "G", "U": "U", "T": "U", "I": "I",
    "DA": "A", "DC": "C", "DG": "G", "DT": "U", "DU": "U",
    "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U", "THY": "U",
    "PSU": "U", "H2U": "U", "5MU": "U", "OMG": "G", "OMC": "C", "OMA": "A",
}

def resname_to_base(resname: str) -> str:
    r = (resname or "").strip().upper()
    if r in BASE_MAP:
        return BASE_MAP[r]
    if len(r) == 1 and r in {"A", "C", "G", "U", "T", "I"}:
        return "U" if r == "T" else r
    return "N"


# ----------------------------
# PDB ダウンロード（ディスクキャッシュ）
# ----------------------------
def load_pdb_text_with_cache(url: str, session: requests.Session, cache_dir: str, timeout: int) -> str:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    fname = cache_path / f"{h}.pdb"

    if fname.exists():
        return fname.read_text(encoding="utf-8", errors="ignore")

    r = session.get(url, timeout=timeout)
    r.raise_for_status()

    if url.lower().endswith(".gz"):
        raw = gzip.decompress(r.content)
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = r.text

    fname.write_text(text, encoding="utf-8")
    return text


# ----------------------------
# PDBファイル名から chain token を抽出
# 例: protein_nucleic-6ahu-1-6ahu_J-1-6ahu_T-1.pdb -> ("J","T")
# ----------------------------
def tokens_from_download_url(pdb_url: str) -> Tuple[str, str]:
    base = (pdb_url.rsplit("/", 1)[-1] if pdb_url else "")
    parts = base.split("-")
    cand = [p for p in parts if "_" in p]
    if len(cand) >= 2:
        t1 = cand[0].split("_")[-1].split(".")[0]
        t2 = cand[1].split("_")[-1].split(".")[0]
        return (t1, t2)
    return ("", "")


# ----------------------------
# chain 推定（chain.id or segid）
# ※segidはバージョン差があるので attribute fallback
# ※segid探索は「最初の1原子だけ」（高速）
# ----------------------------
def chain_first_segid(chain) -> str:
    for atom in chain.get_atoms():
        if hasattr(atom, "get_segid"):
            return (atom.get_segid() or "").strip()
        return (getattr(atom, "segid", "") or "").strip()
    return ""

def pick_chain(structure, expected_token: str, want: str):
    model = next(structure.get_models())
    chains = list(model.get_chains())
    exp = (expected_token or "").strip()

    def count_type(chain) -> int:
        n = 0
        for res in chain.get_residues():
            if want == "protein":
                if Polypeptide.is_aa(res, standard=False):
                    n += 1
            else:
                if Polypeptide.is_aa(res, standard=False):
                    continue
                atom_names = {a.get_name().strip() for a in res.get_atoms()}
                if any(x in atom_names for x in ("P", "C1'", "C1*", "N9", "N1")):
                    n += 1
        return n

    if exp:
        for ch in chains:
            if ch.id.strip() == exp and count_type(ch) > 0:
                return ch

    if exp:
        for ch in chains:
            if chain_first_segid(ch) == exp and count_type(ch) > 0:
                return ch

    scored = [(count_type(ch), ch) for ch in chains]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


# ----------------------------
# RNA残基と代表原子（アンカー）を作る
# 代表原子: P優先（なければC1'など）
# ----------------------------
def build_rna_residues_and_anchors(rna_chain) -> Tuple[List[Any], List[Any], str]:
    residues: List[Any] = []
    anchors: List[Any] = []
    seq_chars: List[str] = []
    prefer = ("P", "C1'", "C1*", "C4'", "C4*", "N9", "N1")

    for res in rna_chain.get_residues():
        if Polypeptide.is_aa(res, standard=False):
            continue

        atom_names = {a.get_name().strip() for a in res.get_atoms()}
        if not any(x in atom_names for x in ("P", "C1'", "C1*", "N9", "N1")):
            continue

        residues.append(res)
        seq_chars.append(resname_to_base(res.get_resname()))

        anchor = None
        for nm in prefer:
            if nm in res:
                anchor = res[nm]
                break
        if anchor is None:
            for a in res.get_atoms():
                el = (a.element or "").strip().upper()
                if el != "H" and not a.get_name().strip().startswith("H"):
                    anchor = a
                    break
        anchors.append(anchor)

    return residues, anchors, "".join(seq_chars)


# ----------------------------
# 接触（距離）からRNA位置の重み（高速）
# - 各ヌクレオチドにつき代表原子1個だけで search
# ----------------------------
def compute_rna_contacts_weights_fast(
    protein_atoms: List[Any],
    rna_anchors: List[Any],
    cutoff: float,
) -> Tuple[List[int], List[float]]:
    if not protein_atoms or not rna_anchors:
        return [], []

    ns = NeighborSearch(protein_atoms)
    L = len(rna_anchors)

    weights = [0.0] * (L + 1)
    pos_list: List[int] = []

    for i, atom in enumerate(rna_anchors, start=1):
        if atom is None:
            continue
        neigh = ns.search(atom.coord, cutoff, level="R")
        if neigh:
            pos_list.append(i)
            weights[i] = float(len(set(neigh)))
    return pos_list, weights


# ----------------------------
# 切り出し
# ----------------------------
@dataclass
class TrimResult:
    trimmed_seq: str
    start_1based: int
    end_1based: int
    mode: str

def best_window_by_weights(weights_1based: List[float], max_len: int) -> Tuple[int, int]:
    L = len(weights_1based) - 1
    if L <= max_len:
        return (1, L)
    cur = sum(weights_1based[1 : max_len + 1])
    best = cur
    best_s = 1
    for s in range(2, L - max_len + 2):
        cur += weights_1based[s + max_len - 1] - weights_1based[s - 1]
        if cur > best:
            best = cur
            best_s = s
    return (best_s, best_s + max_len - 1)

def trim_by_contacts(
    rna_seq: str,
    pos_list: List[int],
    weights: List[float],
    max_len: int,
) -> Optional[TrimResult]:
    L = len(rna_seq)
    if L <= max_len:
        return TrimResult(rna_seq, 1, L, "no_need")

    if not pos_list:
        return None

    pos_list = sorted(p for p in pos_list if 1 <= p <= L)
    if not pos_list:
        return None

    mn, mx = pos_list[0], pos_list[-1]
    span = mx - mn + 1

    if span <= max_len:
        mn2, mx2 = mn, mx
        # 短すぎる場合は左右+5
        if (mx2 - mn2 + 1) <= 10:
            mn2 = max(1, mn2 - 5)
            mx2 = min(L, mx2 + 5)
        trimmed = rna_seq[mn2 - 1 : mx2]
        mode = "contact_span_padded" if (mn2 != mn or mx2 != mx) else "contact_span"
        return TrimResult(trimmed, mn2, mx2, mode)

    if not weights or len(weights) != L + 1:
        w = [0.0] * (L + 1)
        for p in pos_list:
            w[p] = 1.0
        weights = w

    s, e = best_window_by_weights(weights, max_len=max_len)
    trimmed = rna_seq[s - 1 : e]
    return TrimResult(trimmed, s, e, "best_window")


# ----------------------------
# 並列ワーカー（URL 1つ分を処理して結果を返す）
# ※ワーカー内で print しない（I/Oで遅くなる）
# ----------------------------
def _process_one_url(task: dict) -> dict:
    url = task["url"]
    idxs = task["idxs"]               # このURLに紐づく「長いRNA」行index
    seqs = task["seqs"]               # idxsと同順のCSV s2_sequence
    pdb_ids = task["pdb_ids"]         # 表示用
    vis = task["vis"]                 # s2_number_of_visible_residues（あれば）
    max_len = task["max_len"]
    cutoff = task["cutoff"]
    timeout = task["timeout"]
    cache_dir = task["cache_dir"]

    out_updates: Dict[int, Dict[str, Any]] = {}
    out_skips: List[Tuple[int, str]] = []

    session = requests.Session()
    session.headers.update({"User-Agent": "ppi3d-pdb-rna-trimmer/parallel-1.0"})

    try:
        if not (url.lower().endswith(".pdb") or url.lower().endswith(".pdb.gz")):
            for ii in idxs:
                out_skips.append((ii, "not_pdb"))
            return {"url": url, "updates": out_updates, "skips": out_skips, "err": ""}

        pdb_text = load_pdb_text_with_cache(url, session, cache_dir, timeout)

        parser = PDBParser(QUIET=True, PERMISSIVE=True)
        structure = parser.get_structure("x", io.StringIO(pdb_text))

        t1, t2 = tokens_from_download_url(url)
        protein_chain = pick_chain(structure, t1, want="protein")
        rna_chain = pick_chain(structure, t2, want="nucleic")
        if protein_chain is None or rna_chain is None:
            for ii in idxs:
                out_skips.append((ii, "cannot_identify_chain"))
            return {"url": url, "updates": out_updates, "skips": out_skips, "err": ""}

        # protein heavy atoms
        protein_atoms = []
        for res in protein_chain.get_residues():
            if not Polypeptide.is_aa(res, standard=False):
                continue
            for atom in res.get_atoms():
                el = (atom.element or "").strip().upper()
                if el == "H" or atom.get_name().strip().startswith("H"):
                    continue
                protein_atoms.append(atom)

        _, rna_anchors, rna_seq_pdb = build_rna_residues_and_anchors(rna_chain)
        L_pdb = len(rna_seq_pdb)
        if L_pdb == 0:
            for ii in idxs:
                out_skips.append((ii, "rna_not_found"))
            return {"url": url, "updates": out_updates, "skips": out_skips, "err": ""}

        pos_list, weights = compute_rna_contacts_weights_fast(protein_atoms, rna_anchors, cutoff)
        if not pos_list:
            for ii in idxs:
                out_skips.append((ii, "no_contacts"))
            return {"url": url, "updates": out_updates, "skips": out_skips, "err": ""}

        # 同一配列の繰り返しに備え小キャッシュ
        trim_cache: Dict[str, Optional[TrimResult]] = {}

        for ii, rna_seq_csv, pid, v in zip(idxs, seqs, pdb_ids, vis):
            # csvが長い行のみここに来る想定
            # 位置対応の安全のため：長さ一致ならCSV、違うならPDB配列
            use_seq = rna_seq_csv if len(rna_seq_csv) == L_pdb else rna_seq_pdb

            # もしPDB配列がmax_len以下なら、csvが長くても「PDB配列で置換」して救済する
            if len(use_seq) <= max_len:
                out_updates[ii] = {
                    "s2_sequence": use_seq,
                    "s2_number_of_residues": len(use_seq),
                    "s2_number_of_visible_residues": (min(int(v), len(use_seq)) if v is not None else len(use_seq)),
                    "rna_trim_mode": "pdb_short_replace",
                    "rna_trim_start_1based": 1,
                    "rna_trim_end_1based": len(use_seq),
                    "rna_trim_len": len(use_seq),
                }
                continue

            if use_seq in trim_cache:
                tr = trim_cache[use_seq]
            else:
                tr = trim_by_contacts(use_seq, pos_list, weights, max_len)
                trim_cache[use_seq] = tr

            if tr is None:
                out_skips.append((ii, "trim_failed"))
                continue

            out_updates[ii] = {
                "s2_sequence": tr.trimmed_seq,
                "s2_number_of_residues": len(tr.trimmed_seq),
                "s2_number_of_visible_residues": (min(int(v), len(tr.trimmed_seq)) if v is not None else len(tr.trimmed_seq)),
                "rna_trim_mode": tr.mode,
                "rna_trim_start_1based": tr.start_1based,
                "rna_trim_end_1based": tr.end_1based,
                "rna_trim_len": len(tr.trimmed_seq),
            }

        return {"url": url, "updates": out_updates, "skips": out_skips, "err": ""}

    except Exception as e:
        # URL単位で落ちたらそのURLの長い行を全部skip扱い
        msg = f"{type(e).__name__}: {e}"
        for ii in idxs:
            out_skips.append((ii, "exception"))
        return {"url": url, "updates": out_updates, "skips": out_skips, "err": msg}


# ----------------------------
# main（並列化）
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--cutoff", type=float, default=4.5)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--cache_dir", type=str, default="ppi3d_pdb_cache")
    ap.add_argument("--max_workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))))
    ap.add_argument("--chunksize", type=int, default=5, help="ProcessPoolExecutor.mapのchunksize")
    ap.add_argument("--verbose_ok", action="store_true", help="成功ログを出す（遅くなります）")
    ap.add_argument("--verbose_skip", action="store_true", help="skipを行単位で出す（遅くなります）")
    args = ap.parse_args()

    # dtype警告を抑えて安定化
    df = pd.read_csv(args.in_csv, low_memory=False).reset_index(drop=True)

    # 追跡列（なければ追加）
    for c in ["rna_trim_mode", "rna_trim_start_1based", "rna_trim_end_1based", "rna_trim_len"]:
        if c not in df.columns:
            df[c] = pd.NA

    keep = np.ones(len(df), dtype=bool)

    # 長いRNAだけ対象（RNAはs2のみ）
    s2 = df["s2_sequence"].fillna("").astype(str)
    long_mask = s2.str.len().values > args.max_len
    long_idxs_all = np.where(long_mask)[0].tolist()

    # download_urlが無い長い行は即スキップ
    dl = df["download_url"].fillna("").astype(str)
    missing_url_long = [i for i in long_idxs_all if dl.iat[i].strip() == ""]
    if missing_url_long:
        keep[missing_url_long] = False
        if args.verbose_skip:
            for i in missing_url_long:
                print(f"[skip] idx={i} pdb_id={df.at[i,'pdb_id']} : missing download_url (RNA len={len(str(df.at[i,'s2_sequence']))})",
                      file=sys.stderr)
        else:
            print(f"[skip] n={len(missing_url_long)} : missing download_url (long RNA)", file=sys.stderr)

    # URLごとに「長い行」をまとめたタスクリストを作る
    tasks: List[dict] = []
    # groupbyは全行グループになるので、longだけで辞書を作る（軽い）
    url_to_idxs: Dict[str, List[int]] = {}
    for i in long_idxs_all:
        u = dl.iat[i].strip()
        if not u:
            continue
        url_to_idxs.setdefault(u, []).append(i)

    # 可視残基数（無い場合はNone）
    has_vis = "s2_number_of_visible_residues" in df.columns
    for url, idxs in url_to_idxs.items():
        seqs = [str(df.at[i, "s2_sequence"] or "") for i in idxs]
        pdb_ids = [str(df.at[i, "pdb_id"] or "") for i in idxs]
        vis = []
        if has_vis:
            for i in idxs:
                v = df.at[i, "s2_number_of_visible_residues"]
                if pd.isna(v):
                    vis.append(None)
                else:
                    try:
                        vis.append(int(v))
                    except Exception:
                        vis.append(None)
        else:
            vis = [None] * len(idxs)

        tasks.append({
            "url": url,
            "idxs": idxs,
            "seqs": seqs,
            "pdb_ids": pdb_ids,
            "vis": vis,
            "max_len": args.max_len,
            "cutoff": args.cutoff,
            "timeout": args.timeout,
            "cache_dir": args.cache_dir,
        })

    if not tasks:
        out_df = df[keep].copy()
        out_df.to_csv(args.out_csv, index=False)
        print(f"saved: {args.out_csv}  (kept {len(out_df)}/{len(df)})")
        return

    # 並列実行
    from concurrent.futures import ProcessPoolExecutor

    skip_counts: Dict[str, int] = {}
    example_skips: Dict[str, Tuple[int, str]] = {}
    url_level_errors: List[Tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        it = ex.map(_process_one_url, tasks, chunksize=args.chunksize)
        for res in tqdm(it, total=len(tasks), desc="parallel(url)", file=sys.stdout):
            # URL単位の大きな例外メッセージ
            if res.get("err"):
                url_level_errors.append((res["url"], res["err"]))

            # update 反映
            updates: Dict[int, Dict[str, Any]] = res.get("updates", {})
            for idx, up in updates.items():
                df.at[idx, "s2_sequence"] = up["s2_sequence"]
                if "s2_number_of_residues" in df.columns:
                    df.at[idx, "s2_number_of_residues"] = up["s2_number_of_residues"]
                if "s2_number_of_visible_residues" in df.columns:
                    df.at[idx, "s2_number_of_visible_residues"] = up["s2_number_of_visible_residues"]

                df.at[idx, "rna_trim_mode"] = up["rna_trim_mode"]
                df.at[idx, "rna_trim_start_1based"] = up["rna_trim_start_1based"]
                df.at[idx, "rna_trim_end_1based"] = up["rna_trim_end_1based"]
                df.at[idx, "rna_trim_len"] = up["rna_trim_len"]

                if args.verbose_ok:
                    print(f"[ok] idx={idx} pdb_id={df.at[idx,'pdb_id']} : trimmed to {int(up['rna_trim_len'])} nt (mode={up['rna_trim_mode']})")

            # skip 反映（長いRNAのみが対象）
            skips: List[Tuple[int, str]] = res.get("skips", [])
            if skips:
                for idx, reason in skips:
                    keep[idx] = False
                    skip_counts[reason] = skip_counts.get(reason, 0) + 1
                    if reason not in example_skips:
                        example_skips[reason] = (idx, str(df.at[idx, "pdb_id"]))

                    if args.verbose_skip:
                        print(f"[skip] idx={idx} pdb_id={df.at[idx,'pdb_id']} : {reason} (RNA len={len(str(df.at[idx,'s2_sequence']))})",
                              file=sys.stderr)

    # まとめ表示（verbose_skipで無ければここで概況だけ）
    if not args.verbose_skip and skip_counts:
        print("[skip summary]", file=sys.stderr)
        for k, v in sorted(skip_counts.items(), key=lambda x: -x[1]):
            ex_idx, ex_pdb = example_skips.get(k, (-1, ""))
            print(f"  {k}: {v}  (e.g. idx={ex_idx} pdb_id={ex_pdb})", file=sys.stderr)

    if url_level_errors:
        # URL単位エラーは数件だけ表示（多すぎると遅い）
        print("[url-level errors] (first 5)", file=sys.stderr)
        for u, m in url_level_errors[:5]:
            print(f"  {u} : {m}", file=sys.stderr)

    out_df = df[keep].copy()
    out_df.to_csv(args.out_csv, index=False)
    print(f"saved: {args.out_csv}  (kept {len(out_df)}/{len(df)})")


if __name__ == "__main__":
    main()
