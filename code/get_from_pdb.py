#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 依存: pip install rcsb-api requests tqdm

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from rcsbapi.search import search_attributes as attrs

def fetch_protein_rna_entry_ids():
    """Protein も RNA も少なくとも1つ含むエントリのPDB ID一覧を取得"""
    query = (
        attrs.rcsb_entry_info.polymer_entity_count_protein >= 1
    ) & (
        attrs.rcsb_entry_info.polymer_entity_count_RNA >= 1  # ← ここを大文字に
    )
    return list(query())  # 例: ["1ABC", "2DEF", ...]

def download_one(pdb_id: str, outdir: Path, retries: int = 3, timeout: int = 60) -> str:
    pid = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pid}.cif.gz"
    outpath = outdir / f"{pid}.cif.gz"
    if outpath.exists():
        return "skip"

    for i in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(outpath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=512 * 1024):
                        if chunk:
                            f.write(chunk)
            return "ok"
        except Exception:
            time.sleep(1.5 * (i + 1))
    return "fail"

def main(out_dir="mmcif_protein_rna", workers=8):
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    ids = fetch_protein_rna_entry_ids()
    print(f"ヒット数: {len(ids)} 件。保存先: {outdir}")

    status = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, pid, outdir): pid for pid in ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            st = fut.result()
            status[st] += 1

    print(f"完了: OK={status['ok']} / 既存(Skip)={status['skip']} / 失敗={status['fail']}")

if __name__ == "__main__":
    main(out_dir="mmcif_protein_rna", workers=4)
