#!/bin/bash
#$ -l h_vmem=100G

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
num_workers=0

source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate reinforce

cd /home/slab/ishiiayuka/M2

nohup python -u reinforce_ppi3d_AR_offtarget_test.py > output_test_1123_3.log 2> error_test_1123_3.log &