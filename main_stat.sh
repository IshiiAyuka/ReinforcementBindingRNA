#!/bin/bash
#$ -l h_vmem=150G

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
num_workers=0

source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate reinforce

cd /home/slab/ishiiayuka/M2

nohup python -u GC_Length_stat.py > output_1209.log 2> error_1209.log &
