#!/bin/bash
#$ -l h_vmem=100G
source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate esm2

cd /home/slab/ishiiayuka/M2/Decoder

nohup python -u main_train_evaluate.py >main_output_onlytrain.log 2>main_error_onlytrain.log &
pid=$!
