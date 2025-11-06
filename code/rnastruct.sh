#!/bin/bash
#$ -l h_vmem=100G

source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate rnastructure
export DATAPATH="$CONDA_PREFIX/share/rnastructure/data_tables"

# 実行（入力CSV → 出力サマリーCSV）
nohup python -u rnastruct_dynalign.py \
  --csv predictions_after_reinforce.csv \
  --threads 4 \
  > output_structure2.log 2> error_structure2.log &
