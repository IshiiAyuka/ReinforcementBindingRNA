#!/bin/bash
#$ -l h_vmem=100G
source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate esm2

cd /home/slab/ishiiayuka/M2

nohup python -u Decoder/main_DeepCLIP_AR.py >Decoder/output_error/output_1213.log 2>Decoder/output_error/error_1213.log &
pid=$!
