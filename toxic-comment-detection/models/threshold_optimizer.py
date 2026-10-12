import numpy as np
from sklearn.metrics import f1_score
from typing import List, Dict

class ThresholdOptimizer:
    """
    Optimizes decision thresholds per label to maximize Macro F1 score on the validation set.
    """
    def __init__(self, num_labels: int):
        self.num_labels = num_labels
        self.best_thresholds = np.ones(num_labels) * 0.5
        
    def fit(self, y_val_true: np.ndarray, y_val_probs: np.ndarray) -> np.ndarray:
        """
        Finds the optimal threshold for each label using grid search.
        """
        thresholds = np.arange(0.1, 0.9, 0.05)
        
        for i in range(self.num_labels):
            best_f1 = 0
            best_t = 0.5
            
            y_true_col = y_val_true[:, i]
            y_prob_col = y_val_probs[:, i]
            
            for t in thresholds:
                y_pred_col = (y_prob_col >= t).astype(int)
                score = f1_score(y_true_col, y_pred_col, zero_division=0)
                
                if score > best_f1:
                    best_f1 = score
                    best_t = t
                    
            self.best_thresholds[i] = best_t
            
        return self.best_thresholds
        
    def predict(self, y_probs: np.ndarray) -> np.ndarray:
        """
        Applies the optimized thresholds to generate binary predictions.
        """
        y_pred = np.zeros_like(y_probs)
        for i in range(self.num_labels):
            y_pred[:, i] = (y_probs[:, i] >= self.best_thresholds[i]).astype(int)
        return y_pred
