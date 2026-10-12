# Prototype-Augmented Multi-Label Toxic Comment Detection

This repository contains the official PyTorch implementation for the research project:
**Prototype-Augmented Multi-Label Toxic Comment Detection using DeBERTa with Adaptive Prototype Selection and Prototype-Aware Cross Attention**.

This project implements a state-of-the-art framework for classifying toxic comments. Rather than relying purely on parametric models, this architecture leverages a non-parametric **FAISS prototype memory**. The DeBERTa embeddings are enriched via **Adaptive Prototype Selection** and **Prototype-Aware Cross Attention** using retrieved historical comments. Furthermore, we implement **Label Correlation Modeling** to capture dependencies between toxicity types, and **Adaptive Threshold Optimization** to maximize the Macro F1 Score.

## Features & Architecture
- **Pretrained Encoder**: `microsoft/deberta-v3-base`
- **Retrieval Mechanism**: FAISS (Inner Product / L2) for Top-K nearest neighbors.
- **Adaptive Selection**: Learnable module to assign importance to retrieved prototypes.
- **Cross Attention**: Multi-head attention using the comment as Query and Prototypes as Keys/Values.
- **Label Correlation**: Models inter-label dependencies using self-attention.
- **Modularity**: Fully configuration-driven. Disable/Enable FAISS, Attention, or Label Correlation for rigorous ablation studies.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd toxic-comment-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you have a GPU, consider installing `faiss-gpu` instead of `faiss-cpu` for faster retrieval.*

## Dataset Preparation

This project uses the official Kaggle **Jigsaw Toxic Comment Classification Challenge** dataset.

1. Download `train.csv` from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).
2. Place `train.csv` inside `data/raw/train.csv`.
3. Verify and Preprocess:
   ```bash
   python data/download.py
   python data/preprocess.py
   ```
   This will clean the data and split it into `train.csv`, `val.csv`, and `test.csv` inside `data/processed/`.

## Experimental Pipeline

### Phase 1: Train Baseline & Extract Memory
First, you need to train the baseline model (DeBERTa + Linear Head) to get high-quality CLS embeddings.
1. Set `use_faiss_memory: false` in `configs/config.yaml`.
2. Train Baseline:
   ```bash
   python train.py
   ```
3. Once trained, build the FAISS Memory from the training set using the baseline weights:
   ```bash
   python train.py --build_memory
   ```

### Phase 2: Train Full Proposed Model
Now that the prototype memory (`prototype_memory/`) is created:
1. Set `use_faiss_memory: true` and enable all other modules (`use_adaptive_selection`, `use_prototype_attention`, `use_label_correlation`) in `configs/config.yaml`.
2. Train the Full Model:
   ```bash
   python train.py
   ```

## Evaluation & Inference

**Evaluate on Test Set:**
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

**Run Inference on Custom Text:**
This will classify the text, display the retrieved prototypes, and generate an attention heatmap.
```bash
python inference.py --checkpoint checkpoints/best_model.pt --text "You are an absolute idiot and I will find you."
```

## Reproducibility
All random seeds are strictly enforced through `utils/seed.py` to ensure reproducible experiments. The hyperparameter configuration is fully transparent inside `configs/config.yaml`.

## License
MIT License
