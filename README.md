# Prototype-Augmented Multi-Label Toxic Comment Detection

Research-grade PyTorch implementation of **Prototype-Augmented Multi-Label Toxic Comment Detection using DeBERTa with Adaptive Prototype Selection and Prototype-Aware Cross Attention**. The project is designed for reproducible IEEE-style experimentation rather than notebook-only demonstration.

The system predicts six independent Jigsaw labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. Labels are never treated as mutually exclusive; training always uses `BCEWithLogitsLoss`.

## Architecture

```text
Comment → DeBERTa-v3 CLS embedding
        → FAISS cosine retrieval
        → adaptive prototype scoring
        → prototype-aware multi-head cross-attention
        → gated residual fusion
        → learnable label-correlation propagation
        → six logits
        → validation-optimized per-label thresholds
```

FAISS provides a nondifferentiable candidate set. The selector, attention, fusion, correlation model, and classifier are differentiable and jointly optimized. To prevent target leakage, prototype compatibility uses provisional model probabilities rather than the query’s ground-truth labels.

## Repository map

- `configs/`: canonical configuration and ablation overrides.
- `data/`: official archive preparation, validation, splitting, dynamic tokenization, and collation.
- `models/`: baseline encoder/classifier and all proposed modules.
- `retrieval/`: embedding extraction, cosine FAISS indexes, and versioned prototype storage.
- `training/`: BCE objective, training, evaluation, early stopping, and experiment orchestration.
- `utils/`: configuration, metrics, plots, logging, checkpoints, devices, seeds, and result tracking.
- `tests/`: preprocessing, dataset, metric, retrieval, selector, attention, correlation, and threshold tests.
- `train.py`, `build_memory.py`, `evaluate.py`, `inference.py`: primary entry points.

## Installation

Python 3.10 or newer is recommended. Create and activate an isolated environment, then run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build from the official PyTorch instructions when using an NVIDIA GPU. `faiss-cpu` is the portable default. A platform-specific FAISS GPU installation may be substituted without changing the code.

Optional packages:

```powershell
python -m pip install tensorboard umap-learn kaggle
```

- `tensorboard` enables TensorBoard logs; CSV logging always remains active.
- `umap-learn` enables UMAP plots. The code raises a clear error rather than silently substituting t-SNE.
- `kaggle` enables direct competition download when Kaggle credentials are configured.

## Dataset preparation

The uploaded official competition files are already under `data/raw/`. To reproduce extraction from an archive:

```powershell
python -m data.download --archive "C:\path\to\archive.zip" --destination data/raw
```

Prepare deterministic internal train, validation, and test splits:

```powershell
python -m data.preprocess --config configs/config.yaml
```

The script verifies all required columns, binary label values, unique IDs, and nonempty comments; drops rows with missing required values; searches deterministic seeded candidate splits for the closest per-label prevalence; and writes `data/processed/manifest.json` with hashes, sample counts, prevalence, and positive counts.

The competition `test.csv` has hidden labels (`-1` in `test_labels.csv`) and therefore is not used for research metric reporting. The repository creates a held-out test partition from the official labeled `train.csv` without modifying the raw source.

## Training workflow

### 1. Train the DeBERTa baseline

```powershell
python train.py --experiment configs/experiments/baseline.yaml
```

Each run creates `outputs/<experiment>/<timestamp>/` containing:

- `best.pt` and `last.pt` checkpoints;
- resolved configuration;
- training log and epoch-level `history.csv`;
- optional TensorBoard events.

Checkpoints include model, optimizer, scheduler, AMP scaler, epoch, global step, metrics, early-stopping state, full RNG states, configuration, and fitted thresholds when applicable. Set `training.resume_from` in YAML to resume an interrupted run.

### 2. Build prototype memory

```powershell
python build_memory.py --checkpoint outputs/baseline/<run>/best.pt
```

This extracts float32 CLS vectors for every training sample, L2-normalizes them, builds the configured FAISS index, and persists an aligned store containing embeddings, comments, IDs, and all six labels. Self-neighbors are excluded during training by stable sample ID.

### 3. Train the complete model

```powershell
python train.py --experiment configs/experiments/full_model.yaml --initialize-from outputs/baseline/<run>/best.pt
```

The selector scores semantic similarity, provisional-label compatibility, prediction confidence, prototype label density, and learned contextual query–prototype interactions. Hard top-K candidate masking is followed by differentiable soft weighting. Cross-attention uses the comment CLS as query and prototype embeddings as keys and values; a learned gate controls residual fusion.

The label-correlation module creates label-specific states and propagates logits through a row-normalized learnable correlation matrix. A symmetry/diagonal regularizer discourages degenerate graphs without imposing label exclusivity.

## Ablation studies

Available overrides are:

- `baseline.yaml`
- `baseline_faiss.yaml`
- `adaptive_selector.yaml`
- `prototype_attention.yaml`
- `label_correlation.yaml`
- `adaptive_thresholds.yaml`
- `full_model.yaml`

Run the entire pipeline—including preprocessing, baseline training, memory construction, and all ablations—with:

```powershell
python run_experiments.py
```

To reuse an existing baseline:

```powershell
python run_experiments.py --skip-preprocess --baseline-checkpoint path\to\best.pt
```

Every run appends a consistent row to `outputs/experiment_results.csv`. The retrieval-only FAISS control uses similarity-weighted neighbor label evidence as calibrated logit evidence, giving that ablation a defined predictive effect.

## Evaluation and thresholds

```powershell
python evaluate.py `
  --checkpoint outputs/full_model/<run>/best.pt `
  --experiment configs/experiments/full_model.yaml `
  --output outputs/evaluation/full_model
```

Thresholds are fitted independently on validation probabilities to maximize each label’s F1, which maximizes their macro average. They are attached atomically to the checkpoint and saved as JSON. Training remains logits-based BCE; thresholds affect only decision-time evaluation and inference.

Reported metrics include macro/micro F1, precision and recall, per-label and macro ROC-AUC where defined, Hamming loss, and exact match/subset accuracy. Exact match ratio and subset accuracy are logged under both requested names but are mathematically identical.

Evaluation produces predictions, confusion matrices, ROC curves, Precision–Recall curves, and per-label metric plots. `utils.visualization` additionally provides training curves, t-SNE/optional UMAP embeddings, neighbor tables, and attention heatmaps.

## Explainable inference

```powershell
python inference.py `
  --checkpoint outputs/full_model/<run>/best.pt `
  --experiment configs/experiments/full_model.yaml `
  --text "You are awful." `
  --output outputs/inference/example.json
```

The JSON report contains every label confidence and threshold, predicted labels, retrieved comments, cosine similarities, prototype annotations, selector weights, selected status, and cross-attention influence weights. Repeat `--text` for batching or pass `--input-file` with one comment per line.

## Configuration

`configs/config.yaml` controls paths, sequence length, encoder, projection dimension, batch sizes, optimizer rates and betas, weight decay, warmup, gradient accumulation, clipping, AMP, early stopping, retrieval K, selected K, FAISS type (`flat`, `ivf`, or `hnsw`), attention heads, dropout, correlation regularization, threshold grid, seeds, and output locations.

The native DeBERTa hidden size is authoritative. `model.projection_dim` adds a learned projection for retrieval/attention; it does not pretend to alter the pretrained encoder width. Attention dimensions must be divisible by the configured head count.

## Reproducibility

1. Preserve `data/processed/manifest.json` with results.
2. Keep the resolved run configuration and checkpoint together.
3. Use the same Python/library and CUDA versions.
4. Keep `training.seed` and deterministic mode fixed.
5. Build prototype memory from the exact baseline checkpoint reported in the experiment.
6. Fit thresholds only on validation predictions; report test results once.

Deterministic algorithms are requested with warnings because a small number of GPU kernels may lack deterministic implementations. Data-loader generators and workers are seeded explicitly.

## Testing

```powershell
python -m unittest discover -s tests -v
```

Run a syntax-only audit without importing third-party packages:

```powershell
python -m compileall -q .
```

## Export

```powershell
python export_model.py `
  --checkpoint outputs/full_model/<run>/best.pt `
  --experiment configs/experiments/full_model.yaml `
  --output exports/full_model `
  --include-memory
```

The bundle contains the checkpoint, resolved configuration, optional thresholds, optional prototype memory, and a versioned manifest.

## Troubleshooting

- **Out of memory:** reduce `training.batch_size`, `data.max_length`, `retrieval.top_k`, or use gradient accumulation.
- **FAISS import error on Windows:** use a compatible Python version/environment or a Conda FAISS build; `faiss-cpu` support varies by platform release.
- **DeBERTa tokenizer error:** ensure `sentencepiece` is installed and restart the Python process.
- **Kaggle authentication error:** place `kaggle.json` in Kaggle’s expected credentials directory or extract the archive manually.
- **Missing prototype memory:** train the baseline and run `build_memory.py` before prototype-enabled configurations.
- **Checkpoint mismatch:** use the same experiment override for training, evaluation, and inference.
- **Undefined ROC-AUC:** a label with only one target class in a split has no defined ROC-AUC; the metric is recorded as `NaN` instead of fabricated.

## License

MIT. See `LICENSE`.
