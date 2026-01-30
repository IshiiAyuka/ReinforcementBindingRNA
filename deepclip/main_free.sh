#!/bin/bash
#$ -l h_vmem=150G

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
num_workers=0

source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate deepclip

cd /home/slab/ishiiayuka/M2/deepclip

nohup python -u add_deepclip_score_offtarget_mean.py > output_0130.log 2> error_0130.log &