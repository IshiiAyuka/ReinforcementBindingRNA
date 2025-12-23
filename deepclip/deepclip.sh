#!/bin/bash
source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate deepclip

cd /home/slab/ishiiayuka/M2/deepclip

nohup python -u true_false_sequence.py \
    -u RNAcompete.csv \
    -g /home/slab/ishiiayuka/M2/deepclip/generated_rna/rnacompete_top_bottom.csv \
    -w models/RNCMPT \
    --thr 0.75 \
    > output_1223.log \
    2> error_1223.log & 