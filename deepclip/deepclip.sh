#!/bin/bash
source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate deepclip

cd /home/slab/ishiiayuka/M2/deepclip

for filepath in data/*.csv; do
  ID=$(basename "$filepath" .csv)
  echo "Processing ID=${ID} (file: $filepath)"

  nohup python Mydeepclip_multiple.py \
    RNCMPT_sequences.csv \
    "$filepath" \
    models/RNCMPT \
    > result/${ID}_output.log \
    2> result/${ID}_error.log & 
done