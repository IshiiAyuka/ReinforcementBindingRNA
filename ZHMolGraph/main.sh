#!/bin/bash
#$ -l h_vmem=100G

source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate ZHMolGraphPytorch-1.8

cd /home/slab/ishiiayuka/M2/ZHMolGraph

nohup python predict_RPI.py >output.log 2>error.log &
pid=$!
