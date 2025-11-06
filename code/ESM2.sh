#!/bin/bash
#$ -l h_vmem=100G

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
num_workers=0

source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate esm2

cd /home/slab/ishiiayuka/M2

nohup python -u code/ESM2_CSV.py > code/ESM2_output.log 2> code/ESM2_error.log &