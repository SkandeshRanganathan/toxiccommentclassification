import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, hamming_loss, accuracy_score
from typing import Dict, Any

def compute_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes standard multi-label classification metrics.
    
    Args:
        y_true: Ground truth binary labels (N, num_classes)
        y_pred_probs: Predicted probabilities (N, num_classes)
        threshold: Decision threshold to convert probabilities to binary predictions
        
    Returns:
        Dictionary containing all metrics
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    metrics = {}
    
    # 1. Macro and Micro F1 (Primary metrics)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    
    # 2. Precision and Recall
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    # 3. ROC-AUC (Needs probabilities)
    # Wrap in try-except in case a class has only 1 target value in the batch/dataset
    try:
        metrics["roc_auc_macro"] = roc_auc_score(y_true, y_pred_probs, average="macro")
    except ValueError:
        metrics["roc_auc_macro"] = 0.0
        
    # 4. Hamming Loss (Fraction of incorrectly predicted labels)
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)
    
    # 5. Exact Match Ratio (Subset Accuracy)
    metrics["exact_match"] = accuracy_score(y_true, y_pred)
    
    # Per-class F1 for detailed analysis
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, f1 in enumerate(f1_per_class):
        metrics[f"f1_class_{i}"] = f1
        
    return metrics
