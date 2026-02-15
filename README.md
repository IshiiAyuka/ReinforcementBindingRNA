# Protein-Binding RNA Generation with Reinforcement Learning

## Overview

This repository implements a reinforcement learning framework for generating RNA sequences that bind to target proteins.

### Key Features

- Protein representation using **ESM2 embeddings**
- Autoregressive RNA decoder
- Policy gradient–based reinforcement learning
- Reward computation via internal binding score model

---

## Repository

```
https://github.com/IshiiAyuka/ReinforcementBindingRNA
```

---

## Project Structure

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

# Setup

## Clone Repository

```bash
git clone https://github.com/IshiiAyuka/ReinforcementBindingRNA.git
cd ReinforcementBindingRNA
```

## Create Conda Environment

```bash
conda env create -f environment.yml
conda activate reinforce
```

---

# Dataset

## Supervised Training Data (PPI3D-derived)

Supervised learning data are derived from the **PPI3D database**.

Download from:

```
https://bioinformatics.lt/ppi3d/clusters
```

### Download Instructions

1. Open the URL above.
2. In the filter options, check:

```
Protein-nucleic acid interactions
```

3. Download the corresponding clustered dataset.

These data are used to construct protein–nucleic acid interaction training pairs.

---

## Reinforcement Learning Data (SwissProt-derived)

Protein sequences used for reinforcement learning are obtained from **UniProt (SwissProt)**.

Download from:

```
https://www.uniprot.org
```

### Download Instructions

1. Search for proteins of interest.
2. Apply filter:

```
Reviewed (Swiss-Prot)
```

3. Download sequences in FASTA format.

Only **reviewed (manually curated) SwissProt entries** are used to ensure sequence reliability.

---

## Notes on Data Usage

- PPI3D-derived data are used for supervised components (interaction modeling).
- SwissProt-derived data are used for reinforcement learning–based RNA generation.
- Protein sequences must satisfy the ESM2 length constraint (≤ 1022 residues).

The following are the data used in this study:

```
https://drive.google.com/drive/folders/150VlrV9lSkeJFYODzipgd5Gd446fCt0k?usp=drive_link
```

---

# Embedding

## From FASTA

```
python ESM2_fasta.py protein_sequence.fasta output_proteinfeature.pt
```

## From CSV

```
python ESM2_CSV.py protein_RNA_binding_data.csv output_proteinfeature.pt
```

---

# Training

Training is performed using reinforcement learning.

## Standard Execution

```
python Decoder/main.py /path/to/protein_feat.pt /path/to/data.csv /path/to/trained_model.pt /path/to/loss.png
```

Hyperparameters are defined in:

```
Decoder/config.py
```

---

# Reward Function

```
python reinforce.py /path/to/init_weights.pt /path/to/protein_feat.pt /path/to/proteins.fasta /path/to/output_weights.pt

```

# Inference

```

python Decoder/main_inference.py \
  --ckpt /path/to/ckpt.pt \
  --protein_feat_path /path/to/feat.pt \
  --csv_path /path/to/input.csv \
  --output_path /path/to/output.csv

```

# weights

```
The trained weights can be obtained from the following link:

```
https://drive.google.com/drive/folders/1qEZbcafU578iyxtN3Jr0nsxtiJbgnSwA?usp=drive_link
```

```
