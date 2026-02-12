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
├── reinforce.py
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

#### From FASTA

```bash
python ESM2_fasta.py 
```

#### From CSV

```bash
python ESM2_CSV.py -
```

---

## Training

Training is performed using reinforcement learning.

### Standard Execution

```bash
python Decoder/main.py 
```

Hyperparameters are defined in:

```
Decoder/config.py
```

---

## Reward Function

```bash
python reinforce.py 
```

---
