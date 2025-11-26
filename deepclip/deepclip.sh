#!/bin/bash
source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate deepclip

cd /home/slab/ishiiayuka/M2/deepclip

nohup python -u Mydeepclip_multiple.py \
    -u RNAcompete.csv \
    -g /home/slab/ishiiayuka/M2/deepclip/generated_rna/BAnG_result.csv \
    -w models/RNCMPT \
    --thr 0.75 \
    > output_1126.log \
    2> error_1126.log & 