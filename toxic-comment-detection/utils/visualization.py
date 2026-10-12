import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
import umap

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list, output_dir: str):
    """Plots and saves confusion matrices for each label."""
    os.makedirs(output_dir, exist_ok=True)
    num_labels = len(labels)
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_labels):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt="d", ax=axes[i], cmap="Blues", cbar=False)
        axes[i].set_title(f"Confusion Matrix: {labels[i]}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"))
    plt.close()

def plot_attention_heatmap(attention_weights: np.ndarray, comment: str, prototype_texts: list, output_path: str):
    """Plots attention weights showing how much the comment attended to each prototype."""
    avg_attention = np.mean(attention_weights[0, :, 0, :], axis=0)
    
    plt.figure(figsize=(10, 6))
    trunc_texts = [text[:50] + "..." if len(text) > 50 else text for text in prototype_texts]
    
    sns.barplot(x=avg_attention, y=trunc_texts, palette="viridis")
    plt.title(f"Attention Weights for input: '{comment[:30]}...'")
    plt.xlabel("Attention Score")
    plt.ylabel("Retrieved Prototypes")
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()

def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, labels: list, output_dir: str):
    """Plots ROC curves for all labels."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per Label")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()

def plot_pr_curves(y_true: np.ndarray, y_probs: np.ndarray, labels: list, output_dir: str):
    """Plots Precision-Recall curves for all labels."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    for i, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        ap = average_precision_score(y_true[:, i], y_probs[:, i])
        plt.plot(recall, precision, label=f"{label} (AP = {ap:.2f})")
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves per Label")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curves.png"))
    plt.close()

def plot_training_curves(history: dict, output_dir: str):
    """Plots training and validation loss/F1 curves."""
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label="Train Loss")
    plt.plot(epochs, history['val_loss'], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_f1_macro'], label="Val Macro F1", color="green")
    plt.title("Validation F1 Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()

def plot_embeddings(embeddings: np.ndarray, labels: np.ndarray, label_names: list, output_dir: str, method="tsne"):
    """Visualizes embeddings using t-SNE or UMAP."""
    os.makedirs(output_dir, exist_ok=True)
    
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
        
    reduced_embs = reducer.fit_transform(embeddings)
    
    # Plotting is tricky for multi-label. We can plot a dominant label or color by toxic vs non-toxic
    is_toxic = np.any(labels > 0, axis=1)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_embs[:, 0], y=reduced_embs[:, 1], hue=is_toxic, palette={True: "red", False: "blue"}, alpha=0.6)
    plt.title(f"{method.upper()} Projection of Comment Embeddings (Toxic vs Non-Toxic)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"embeddings_{method.lower()}.png"))
    plt.close()
