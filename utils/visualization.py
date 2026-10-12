"""Publication-oriented visual diagnostics for multi-label experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve, roc_curve


def _prepare(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def plot_training_curves(history: pd.DataFrame, path: str | Path) -> None:
    columns = [column for column in ("train_loss", "loss", "macro_f1", "micro_f1") if column in history]
    figure, axes = plt.subplots(len(columns), 1, figsize=(8, max(3, 2.8 * len(columns))), squeeze=False)
    for axis, column in zip(axes.ravel(), columns, strict=True):
        axis.plot(history["epoch"], history[column], marker="o")
        axis.set(xlabel="Epoch", ylabel=column.replace("_", " ").title())
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(_prepare(path), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrices(targets: np.ndarray, predictions: np.ndarray, labels: Sequence[str], path: str | Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for index, (axis, label) in enumerate(zip(axes.ravel(), labels, strict=True)):
        ConfusionMatrixDisplay.from_predictions(targets[:, index], predictions[:, index], ax=axis, colorbar=False)
        axis.set_title(label)
    figure.tight_layout()
    figure.savefig(_prepare(path), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_roc_curves(targets: np.ndarray, probabilities: np.ndarray, labels: Sequence[str], path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    for index, label in enumerate(labels):
        if np.unique(targets[:, index]).size < 2:
            continue
        false_positive, true_positive, _ = roc_curve(targets[:, index], probabilities[:, index])
        axis.plot(false_positive, true_positive, label=f"{label} (AUC={auc(false_positive, true_positive):.3f})")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Per-label ROC curves")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(_prepare(path), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_precision_recall_curves(targets: np.ndarray, probabilities: np.ndarray, labels: Sequence[str], path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    for index, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(targets[:, index], probabilities[:, index])
        axis.plot(recall, precision, label=f"{label} (AUC={auc(recall, precision):.3f})")
    axis.set(xlabel="Recall", ylabel="Precision", title="Per-label Precision–Recall curves")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(_prepare(path), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_per_label_metrics(metrics: dict[str, float], labels: Sequence[str], path: str | Path) -> None:
    rows = [{"label": label, "precision": metrics[f"{label}_precision"], "recall": metrics[f"{label}_recall"], "f1": metrics[f"{label}_f1"]} for label in labels]
    long = pd.DataFrame(rows).melt("label", var_name="metric", value_name="score")
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long, x="label", y="score", hue="metric", ax=axis)
    axis.tick_params(axis="x", rotation=25)
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(_prepare(path), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_embeddings(embeddings: np.ndarray, labels: np.ndarray, path: str | Path, method: str = "tsne", seed: int = 42, max_points: int = 5000) -> None:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(embeddings), min(max_points, len(embeddings)), replace=False)
    values, color = embeddings[indices], labels[indices].sum(axis=1)
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
    elif method.lower() == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("UMAP visualization requires the optional `umap-learn` package") from exc
        reducer = umap.UMAP(n_components=2, random_state=seed)
    else:
        raise ValueError("method must be 'tsne' or 'umap'")
    projected = reducer.fit_transform(values)
    figure, axis = plt.subplots(figsize=(8, 7))
    scatter = axis.scatter(projected[:, 0], projected[:, 1], c=color, cmap="viridis", s=8, alpha=0.7)
    figure.colorbar(scatter, ax=axis, label="Number of positive labels")
    axis.set_title(f"Prototype embeddings ({method.upper()})")
    figure.tight_layout()
    figure.savefig(_prepare(path), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_attention_heatmap(weights: np.ndarray, prototype_labels: Sequence[str], path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(max(8, len(prototype_labels) * 0.7), 2.8))
    sns.heatmap(weights.reshape(1, -1), xticklabels=prototype_labels, yticklabels=["query"], cmap="mako", annot=True, fmt=".3f", ax=axis)
    axis.set_title("Prototype influence weights")
    figure.tight_layout()
    figure.savefig(_prepare(path), dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_neighbor_table(rows: Sequence[dict[str, object]], path: str | Path) -> None:
    pd.DataFrame(rows).to_csv(_prepare(path), index=False)

