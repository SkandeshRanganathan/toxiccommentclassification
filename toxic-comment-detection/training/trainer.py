import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import os
import json

from training.evaluator import Evaluator
from utils.checkpoint import save_checkpoint
from utils.logger import setup_logger

logger = setup_logger(__name__)

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        self.epochs = config["training"]["epochs"]
        self.mixed_precision = config["training"].get("mixed_precision", False)
        self.max_grad_norm = config["training"].get("max_grad_norm", 1.0)
        self.patience = config["training"].get("patience", 3)
        
        # Setup scheduler
        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * config["training"].get("warmup_ratio", 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        self.scaler = GradScaler() if self.mixed_precision else None
        self.evaluator = Evaluator(model, val_loader, criterion, device, config)
        
    def train(self):
        best_val_f1 = 0.0
        epochs_no_improve = 0
        
        history = {'train_loss': [], 'val_loss': [], 'val_f1_macro': []}
        
        for epoch in range(1, self.epochs + 1):
            logger.info(f"--- Epoch {epoch}/{self.epochs} ---")
            
            # Training Phase
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc="Training")
            for batch in pbar:
                self.optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids', None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                targets = batch['labels'].to(self.device)
                
                # Mixed precision
                if self.mixed_precision:
                    # Disable autocast explicitly if not on CUDA to prevent Half vs Float errors
                    with autocast(enabled=(self.device.type == 'cuda' and self.mixed_precision)):
                        outputs = self.model(input_ids, attention_mask, token_type_ids)
                        logits = outputs["logits"]
                        loss = self.criterion(logits, targets)
                        
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(input_ids, attention_mask, token_type_ids)
                    logits = outputs["logits"]
                    loss = self.criterion(logits, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                self.scheduler.step()
                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_train_loss = train_loss / len(self.train_loader)
            logger.info(f"Average Training Loss: {avg_train_loss:.4f}")
            
            # Evaluation Phase
            val_loss, metrics = self.evaluator.evaluate()
            val_f1 = metrics["f1_macro"]
            
            logger.info(f"Validation Loss: {val_loss:.4f}")
            logger.info(f"Validation Macro F1: {val_f1:.4f}")
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_f1_macro'].append(val_f1)
            
            # Checkpointing
            is_best = val_f1 > best_val_f1
            if is_best:
                best_val_f1 = val_f1
                epochs_no_improve = 0
                logger.info("New best model found!")
            else:
                epochs_no_improve += 1
                
            save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, val_loss, self.config, is_best)
            
            if epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
                
        # Save training history
        import pandas as pd
        os.makedirs(self.config["project"]["output_dir"], exist_ok=True)
        
        # Save as JSON
        with open(os.path.join(self.config["project"]["output_dir"], "history.json"), "w") as f:
            json.dump(history, f, indent=4)
            
        # Save as structured CSV for experiments
        history_df = pd.DataFrame(history)
        history_df.index = np.arange(1, len(history_df) + 1)
        history_df.index.name = 'epoch'
        history_df.to_csv(os.path.join(self.config["project"]["output_dir"], "training_history.csv"))
            
        return history
