#!/bin/bash
#$ -l h_vmem=100G

source /home/slab/ishiiayuka/.pyenv/versions/anaconda3-2023.03/bin/activate
conda activate getfrompdb

cd /home/slab/ishiiayuka/M2

nohup python -u count_contact.py > output_pdb.log 2> error_pdb.log &