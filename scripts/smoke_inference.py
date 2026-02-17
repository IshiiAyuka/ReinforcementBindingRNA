#!/usr/bin/env python3
"""
Minimal sanity check for Decoder/main_inference.py.

Creates a tiny dummy dataset + dummy checkpoint under tmp_smoke/ and runs inference.
The generated sequence is arbitrary because the checkpoint is randomly initialized.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="tmp_smoke", help="Output directory to write dummy inputs/outputs.")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Local imports from Decoder/
    sys.path.insert(0, os.path.join(repo_root, "Decoder"))
    import torch
    import pandas as pd
    import config
    from model import ProteinToRNA

    feat_path = os.path.join(out_dir, "feat.pt")
    csv_path = os.path.join(out_dir, "input.csv")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    out_csv = os.path.join(out_dir, "out.csv")

    torch.manual_seed(0)
    torch.save({"P12345": torch.randn(7, config.input_dim)}, feat_path)

    pd.DataFrame(
        [
            {
                "subunit_1": "P12345",
                "subunit_2": "X1",
                "s1_sequence": "M" * 20,
                "s2_sequence": "A" * 10,
                "s1_binding_site_cluster_data_40_area": "c1_x",
            }
        ]
    ).to_csv(csv_path, index=False)

    torch.save(
        ProteinToRNA(input_dim=config.input_dim, num_layers=config.num_layers).state_dict(),
        ckpt_path,
    )

    cmd = [
        sys.executable,
        os.path.join(repo_root, "Decoder", "main_inference.py"),
        "--ckpt",
        ckpt_path,
        "--protein_feat_path",
        feat_path,
        "--csv_path",
        csv_path,
        "--output_path",
        out_csv,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("Wrote:", out_csv, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

