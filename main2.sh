#!/bin/bash
#$ -l h_vmem=150G

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
num_workers=0

source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate reinforce

cd /home/slab/ishiiayuka/M2

nohup python -u RNAkiridashi.py \
  --in_csv ppi3d_1128.csv --out_csv ppi3d_1128_.csv \
  --max_len 100 --cutoff 4.5 \
  --cache_dir pdb_cache \
  --max_workers 8 --chunksize 5 \
  > output.log \
  2> error.log &