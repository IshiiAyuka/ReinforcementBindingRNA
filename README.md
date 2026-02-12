# Protein-Binding RNA Generation with Reinforcement Learning

## Overview

This repository implements a reinforcement learning framework for generating RNA sequences that bind to target proteins.

### Key Features

- Protein representation using **ESM2 embeddings**
- Autoregressive RNA decoder
- Policy gradient–based reinforcement learning
- Reward computation via internal binding score model

---

## Repository URL

This repository:

```
https://github.com/IshiiAyuka/ReinforcementBindingRNA
```

---

## Directory Structure

```
.
├── Decoder/
│   ├── config.py
│   ├── dataset.py
│   ├── decode.py
│   ├── evaluate.py
│   ├── main.py
│   ├── model.py
│   ├── plots.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
│
├── ESM2_CSV.py
├── ESM2_fasta.py
├── reinforce_swissprot_AR_offtarget...
├── environment.yml
├── .gitignore
└── README.md
```

---

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/IshiiAyuka/ReinforcementBindingRNA.git
cd ReinforcementBindingRNA
```

---

### 2. Create Conda Environment from environment.yml

```bash
conda env create -f environment.yml
conda activate reinforce
```


---

## Data Preparation

### 1. Prepare Protein Sequences

Prepare protein sequences in FASTA or CSV format.

Example FASTA:

```
>protein_name
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRK...
```

---

### 2. Generate ESM2 Embeddings

#### From FASTA

```bash
python ESM2_fasta.py --input protein.fasta --output protein_embedding.pt
```

#### From CSV

```bash
python ESM2_CSV.py --input proteins.csv --output embeddings.pt
```

If a protein complex contains multiple chains:

- Each chain embedding (320-dim) is computed
- Embeddings are concatenated before being passed to the decoder

---

## Training

Training is performed using reinforcement learning.

### Standard Execution

```bash
nohup python -u Decoder/train.py > output.log 2> error.log &
```

### Monitor Training

```bash
tail -f output.log
```

---

### Stop Training

```bash
ps aux | grep train.py
kill <PID>
```

---

## Using `main.py` (Alternative Entry)

```bash
nohup python -u Decoder/main.py > main_output.log 2> main_error.log &
```

Hyperparameters are defined in:

```
Decoder/config.py
```

---

## Inference (RNA Sequence Decoding)

```bash
nohup python -u Decoder/decode.py > decode.log 2> decode_error.log &
```

Generated sequences are saved to the output location defined in `decode.py`.

---

## Evaluation

```bash
nohup python -u Decoder/evaluate.py > eval.log 2> eval_error.log &
```

### Plotting

```bash
nohup python -u Decoder/plots.py > plots.log 2> plots_error.log &
```

---

## Reward Function

Reward computation is implemented in:

```
Decoder/predict.py
```

Typical reward components include:

- Predicted binding score
- Length-normalized free energy
- Entropy regularization

Policy gradient loss is computed in:

```
Decoder/train.py
```

---

## GPU Usage

Specify GPU for training:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Check CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Reproducibility

To ensure reproducibility, set random seeds inside `train.py`:

```python
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

---

## Notes

- `predict.py` is designed to be called directly inside Python (no subprocess).
- Multi-chain protein complexes are supported via embedding concatenation.
- Designed for large-scale background execution.

---

## Citation

If you use this code in academic research, please cite:

```
(To be added)
```

---

## Author

Ayuka Ishii
