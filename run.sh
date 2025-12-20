#!/bin/bash
#$ -l h_vmem=100G
source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate reinforce

cd /home/slab/ishiiayuka/M2

# RNAcompete top/bottom extractor runner
DOWNLOAD_URL=""
EXTRACT_DIR="/home/slab/ishiiayuka/M2/deepclip/data"
OUTPUT_CSV="/home/slab/ishiiayuka/M2/deepclip/rnacompete_top_bottom.csv"
PROTEIN_LIST_CSV="/home/slab/ishiiayuka/M2/RNAcompete.csv"
DESIGN_FILE="/home/slab/ishiiayuka/M2/RBD_v3_design_unified.txt"
NORM_DATA="/home/slab/ishiiayuka/M2/norm_data.txt.gz"
TOP_N=100
BOTTOM_N=100

if [[ ! -f "$NORM_DATA" ]]; then
  mkdir -p "$EXTRACT_DIR"
  if [[ -z "$DOWNLOAD_URL" ]] && ! ls "$EXTRACT_DIR"/*.csv "$EXTRACT_DIR"/*.tsv "$EXTRACT_DIR"/*.txt >/dev/null 2>&1; then
    echo "[ERROR] $NORM_DATA が見つかりません。norm_data.txt(.gz) を配置するか、DOWNLOAD_URL を設定してください。"
    exit 1
  fi
fi

cmd=(python -u /home/slab/ishiiayuka/M2/rnacompete_top_bottom.py
  --extract-dir "$EXTRACT_DIR"
  --output "$OUTPUT_CSV"
  --protein-list-csv "$PROTEIN_LIST_CSV"
  --design-file "$DESIGN_FILE"
  --norm-data "$NORM_DATA"
  --top "$TOP_N"
  --bottom "$BOTTOM_N"
)

if [[ -n "$DOWNLOAD_URL" ]]; then
  cmd+=(--download-url "$DOWNLOAD_URL")
fi

nohup "${cmd[@]}" > /home/slab/ishiiayuka/M2/output_rnacompete_top_bottom.log 2> /home/slab/ishiiayuka/M2/error_rnacompete_top_bottom.log &
pid=$!
echo "Started (pid=$pid)"
