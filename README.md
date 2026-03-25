# Duplicate Hotel Detection (Entity Resolution)

This repository contains the code and experimental artifacts for the dissertation **“Leveraging Fine-Tuned Generative AI: A Comparative Approach to Identifying Duplicate Hotel Listings Across Multiple Platforms”** (Tanmay Desai, QMUL).

It compares two approaches for hotel-listing deduplication (entity matching):

- **BertJointHead (hybrid BERT)**: a BERT pair encoder whose pooled embedding is augmented with a **normalized Levenshtein similarity** scalar, and trained with **joint heads** for (1) match/no-match and (2) cluster/entity-ID prediction.
- **Meta-Llama-3-8B-Instruct + LoRA**: an instruction-following **generative** formulation that outputs exactly **`MATCH`** / **`NO_MATCH`**, fine-tuned using **parameter-efficient LoRA** (with optional **4-bit quantization**).

Key paper results (from the dissertation PDF):
- **BertJointHead**: F1 **0.8721** (Precision 0.8822, Recall 0.8623)
- **LLaMA + LoRA**: F1 **0.9009** (Precision 0.8621, Recall 0.9434), Accuracy **0.9486**

## Repository layout

- **`hotel_data_duplication/`**: main training/evaluation code used in the dissertation  
  - **`hotel_data_duplication/configs/config.yaml`**: central config (paths + hyperparameters)
  - **`hotel_data_duplication/scripts/train.py`**: BERT JointHead training
  - **`hotel_data_duplication/scripts/evaluate.py`**: BERT evaluation (includes Haversine geo-filtering, threshold sweep, confusion matrix)
  - **`hotel_data_duplication/scripts/llama_finetune.py`**: LLaMA LoRA fine-tuning (instruction prompts, PEFT, optional bitsandbytes quantization)
  - **`hotel_data_duplication/scripts/llamaEvaluate.py`**: LLaMA evaluation (exhaustive pairwise comparison + geo-filter)
  - **`hotel_data_duplication/data/`**: prepared datasets and evaluation artifacts (CSV/JSON outputs)
- **`kaggle/`**: raw Kaggle exports and small helper scripts  
  - Large CSVs are stored via **Git LFS** (see “Large files / Git LFS” below).
- **`merged_matches/`**, **`separated_scores/`**, **`non_matches/`**: intermediate CSVs and utilities used while generating / inspecting matched pairs.
- **`newDataset2Python/`**, **`ukDataset/`**: scripts and datasets used to construct/enrich UK hotel records (incl. OSM/Overpass processing).

## Setup

This is a research codebase (not packaged as a library). Use a Python virtual environment and install dependencies with pip.

### Python

- Recommended: **Python 3.10+**

### Install dependencies

At minimum, the core scripts use:
- `torch`
- `transformers`
- `peft`
- `bitsandbytes` (optional, for 4-bit quantization)
- `pandas`, `numpy`, `scikit-learn`
- `pyyaml`, `tqdm`
- `python-Levenshtein`
- `matplotlib` (plots in BERT evaluation)

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers peft bitsandbytes pandas numpy scikit-learn pyyaml tqdm python-Levenshtein matplotlib
```

## Data

The dissertation builds a **balanced hotel-pair dataset (~7.3k labeled pairs)** and evaluates models with **exhaustive pairwise comparisons** plus **Haversine distance filtering** (typical threshold: **500m**).

Important data files (see `hotel_data_duplication/data/`):
- **`hotel_pairs.csv`**: labeled training pairs (`text_a`, `text_b`, `label`)
- **`hotel_pairs_with_clusters.csv`** / `*_with_clusters.csv`: pairs with `clusterA` / `clusterB` for joint entity-ID training
- **`addresses_corrected_*.csv`**: address records used for evaluation
- **`inference_labels*.csv`**: ground-truth duplicate pairs for evaluation

## Running the experiments

All scripts read configuration from:
- `hotel_data_duplication/configs/config.yaml`

### 1) Train the BERT JointHead model

```bash
python hotel_data_duplication/scripts/train.py
```

Notes:
- `train.py` defines `BertJointHead` and computes Levenshtein similarity via `Levenshtein.ratio(...)`.
- The code infers the number of entity clusters from `clusterA/clusterB` in the training CSV.

### 2) Evaluate the BERT model (threshold sweep + geo-filtering)

```bash
python hotel_data_duplication/scripts/evaluate.py
```

What it does (high level):
- Generates candidate pairs with **Haversine filtering** (distance threshold from config).
- Computes match probabilities and runs a **precision/recall/F1 sweep** to pick an operating threshold.
- Writes evaluation artifacts (threshold sweep, wrong predictions, confusion matrix plot) to the configured output directory.

### 3) Fine-tune LLaMA-3 with LoRA (instruction MATCH/NO_MATCH)

```bash
python hotel_data_duplication/scripts/llama_finetune.py
```

Notes:
- Uses an instruction prompt and trains only on the assistant answer tokens (`MATCH` / `NO_MATCH`).
- Supports **bitsandbytes 4-bit quantization** when installed to reduce VRAM usage.
- Requires access to the base model specified in config (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`).

### 4) Evaluate the LLaMA LoRA model (exhaustive + geo-filter)

```bash
python hotel_data_duplication/scripts/llamaEvaluate.py
```

## Large files / Git LFS

This repo uses **Git LFS** for large Kaggle CSVs in `kaggle/*.csv`.

If you clone this repo, install and pull LFS objects:

```bash
git lfs install
git lfs pull
```

## Dissertation / citation

If you use or build on this code, please cite the dissertation:

> Tanmay Desai. “Leveraging Fine-Tuned Generative AI: A Comparative Approach to Identifying Duplicate Hotel Listings Across Multiple Platforms.” MSc Computer Science Dissertation, Queen Mary University of London.
