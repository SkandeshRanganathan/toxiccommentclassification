import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import os
import json

from utils.logger import setup_logger
from data.dataset import ToxicCommentDataset
from models.full_model import FullToxicModel
from models.baseline_classifier import BaselineClassifier
from training.evaluator import Evaluator
from training.losses import MultiLabelLoss
from utils.checkpoint import load_checkpoint
from models.threshold_optimizer import ThresholdOptimizer
from utils.metrics import compute_metrics
import numpy as np

logger = setup_logger(__name__)

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder_name"])
    
    processed_dir = config["data"]["processed_dir"]
    test_dataset = ToxicCommentDataset(
        data_path=os.path.join(processed_dir, "test.csv"),
        tokenizer=tokenizer,
        text_col=config["data"]["text_col"],
        label_cols=config["data"]["label_cols"],
        max_seq_length=config["data"]["max_seq_length"]
    )
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["eval_batch_size"], shuffle=False)
    
    # Load Model
    if config["model"]["use_faiss_memory"]:
        model = FullToxicModel(config, device).to(device)
    else:
        model = BaselineClassifier(
            encoder_name=config["model"]["encoder_name"],
            embedding_dim=config["model"]["embedding_dim"],
            num_labels=config["model"]["num_labels"]
        ).to(device)
        
    load_checkpoint(args.checkpoint, model)
    criterion = MultiLabelLoss()
    
    evaluator = Evaluator(model, test_loader, criterion, device, config)
    
    # Evaluate with default threshold
    logger.info("Evaluating with default threshold (0.5)...")
    loss, metrics = evaluator.evaluate(threshold=0.5)
    
    logger.info("Test Metrics (Default):")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
        
    # Optional: Threshold Optimization
    if config["evaluation"].get("optimize_thresholds", True):
        logger.info("Optimizing thresholds on validation set...")
        val_dataset = ToxicCommentDataset(
            data_path=os.path.join(processed_dir, "val.csv"),
            tokenizer=tokenizer,
            text_col=config["data"]["text_col"],
            label_cols=config["data"]["label_cols"],
            max_seq_length=config["data"]["max_seq_length"]
        )
        val_loader = DataLoader(val_dataset, batch_size=config["training"]["eval_batch_size"], shuffle=False)
        val_evaluator = Evaluator(model, val_loader, criterion, device, config)
        
        # Gather val probabilities
        model.eval()
        val_targets, val_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                targets = batch['labels'].numpy()
                logits = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch.get('token_type_ids', None)
                )["logits"]
                probs = torch.sigmoid(logits).cpu().numpy()
                val_targets.append(targets)
                val_probs.append(probs)
                
        val_targets = np.vstack(val_targets)
        val_probs = np.vstack(val_probs)
        
        optimizer = ThresholdOptimizer(num_labels=config["model"]["num_labels"])
        best_thresholds = optimizer.fit(val_targets, val_probs)
        logger.info(f"Optimized Thresholds: {best_thresholds}")
        
        # Apply to test set
        test_targets, test_probs = [], []
        with torch.no_grad():
            for batch in test_loader:
                targets = batch['labels'].numpy()
                logits = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch.get('token_type_ids', None)
                )["logits"]
                probs = torch.sigmoid(logits).cpu().numpy()
                test_targets.append(targets)
                test_probs.append(probs)
                
        test_targets = np.vstack(test_targets)
        test_probs = np.vstack(test_probs)
        
        y_pred = optimizer.predict(test_probs)
        
        from sklearn.metrics import f1_score, precision_score, recall_score
        opt_macro_f1 = f1_score(test_targets, y_pred, average="macro", zero_division=0)
        logger.info(f"Test Macro F1 (Optimized Thresholds): {opt_macro_f1:.4f}")
        
        # Save metrics
        metrics["optimized_macro_f1"] = opt_macro_f1
        with open(os.path.join(config["project"]["output_dir"], "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
            
        import pandas as pd
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(config["project"]["output_dir"], "experiment_results.csv"), index=False)

if __name__ == "__main__":
    main()
