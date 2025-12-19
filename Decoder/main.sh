#!/bin/bash
#$ -l h_vmem=100G
source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate reinforce

export RNAFOLD_BIN=/path/to/RNAfold

cd /home/slab/ishiiayuka/M2

nohup python -u Decoder/main_only_predict.py >Decoder/output_1219.log 2>Decoder/error_1219.log &
pid=$!
