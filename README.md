# 🧬 Protein-Binding RNA Generation with Reinforcement Learning

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-required-ee4c2c)
![ESM2](https://img.shields.io/badge/ESM2-embeddings-2b9348)

This repository implements a reinforcement learning framework for generating RNA sequences that bind to target proteins.

## 🚀 Quickstart (Inference First)

Run RNA generation with a trained checkpoint and export results to a CSV.

### 1) ⚙️ Setup

```bash
git clone https://github.com/IshiiAyuka/ReinforcementBindingRNA.git
cd ReinforcementBindingRNA

conda env create -f environment.yml
conda activate reinforce
```

Environment sanity check:

```bash
python -c "import torch, pandas, tqdm; print('ok')"
```

If you do not use conda, you can run with system Python as well:

```bash
# Install PyTorch first (CPU/CUDA): https://pytorch.org/get-started/locally/
/usr/bin/python3 -m pip install --user -r requirements.txt
/usr/bin/python3 -c "import torch, pandas, tqdm, Bio, esm; print('ok')"
```

### 2) 📦 Download trained weights

The trained weights can be obtained from the following link:

```
https://drive.google.com/drive/folders/1qEZbcafU578iyxtN3Jr0nsxtiJbgnSwA?usp=drive_link
```

### 3) 🧠 Prepare protein embeddings (ESM2)

Protein sequences must satisfy the ESM2 length constraint (<= 1022 residues).

From FASTA (SwissProt FASTA etc.):

```bash
python ESM2_fasta.py protein_sequence.fasta output_proteinfeature.pt
```

From CSV (expects columns `file_name` and `sequence`):

```bash
python ESM2_CSV.py protein_RNA_binding_data.csv output_proteinfeature.pt
```

### 4) 🔮 Run inference

Run inference (writes a CSV to `--output_path`):

```bash
python Decoder/main_inference.py \
  --ckpt /path/to/ckpt.pt \
  --protein_feat_path /path/to/feat.pt \
  --csv_path /path/to/input.csv \
  --output_path /path/to/output.csv
```

Run from the repository root (`ReinforcementBindingRNA/`). See below for I/O examples and a minimal sanity check.

### 🧾 I/O Examples (Inference)

Inputs:

- `input.csv` (PPI3D-derived) must include: `subunit_1`, `subunit_2`, `s1_sequence`, `s2_sequence`, `s1_binding_site_cluster_data_40_area`
- `feat.pt` is a dict: `{subunit_1 (str) -> FloatTensor[L, 640]}` saved by `torch.save(...)`

Output:

- `output.csv` columns: `complex_id`, `protein_sequence`, `generated_rna_sequence`, `true_rna_sequence`

Minimal sanity check (no trained weights / no real data):

```bash
python scripts/smoke_inference.py
```

## ✨ Overview

### Key Features

- Protein representation using **ESM2 embeddings**
- Autoregressive RNA decoder
- Policy gradient-based reinforcement learning
- Reward computation via internal binding score model

## Repository

```
https://github.com/IshiiAyuka/ReinforcementBindingRNA
```

## Project Structure

```
.
├── Decoder/
│   ├── config.py
│   ├── dataset.py
│   ├── decode.py
│   ├── evaluate.py
│   ├── main.py
│   ├── main_inference.py
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

## 📚 Dataset

### Supervised Training Data (PPI3D-derived)

Supervised learning data are derived from the **PPI3D database**.

Download: `https://bioinformatics.lt/ppi3d/clusters`

Download instructions:

1. Open the URL above.
2. In the filter options, check `Protein-nucleic acid interactions`.
3. Download the corresponding clustered dataset.

Usage: These data are used to construct protein-nucleic acid interaction training pairs.

### Reinforcement Learning Data (SwissProt-derived)

Protein sequences used for reinforcement learning are obtained from **UniProt (SwissProt)**.

Download: `https://www.uniprot.org`

Download instructions:

1. Search for proteins of interest.
2. Apply filter `Reviewed (Swiss-Prot)`.
3. Download sequences in FASTA format.

Only **reviewed (manually curated) SwissProt entries** are used to ensure sequence reliability.

### Notes on Data Usage

- PPI3D-derived data are used for supervised components (interaction modeling).
- SwissProt-derived data are used for reinforcement learning-based RNA generation.

The following are the data used in this study:

```
https://drive.google.com/drive/folders/150VlrV9lSkeJFYODzipgd5Gd446fCt0k?usp=drive_link
```

## 🏋️ Training

### Supervised training (autoregressive decoder)

```bash
python Decoder/main.py \
  /path/to/protein_feat.pt \
  /path/to/data.csv \
  /path/to/trained_model.pt \
  /path/to/loss.png
```

Hyperparameters are defined in `Decoder/config.py`.

### Reinforcement learning (REINFORCE)

This step depends on LucaOneTasks (reward model runner). From the repository root:

```bash
git clone https://github.com/LucaOne/LucaOneTasks.git LucaOneTasks
ln -s LucaOneTasks/src src
```

If you use pip, install extra dependencies:

```bash
python -m pip install -r requirements-reinforce.txt
```

```bash
python reinforce.py \
  /path/to/init_weights.pt \
  /path/to/protein_feat.pt \
  /path/to/proteins.fasta \
  /path/to/output_weights.pt
```
