import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple

from utils.metrics import compute_metrics
from utils.logger import setup_logger

logger = setup_logger(__name__)

class Evaluator:
    def __init__(self, model, dataloader, criterion, device, config):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.config = config
        
    def evaluate(self, threshold: float = 0.5) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the model on the provided dataloader.
        """
        self.model.eval()
        total_loss = 0.0
        
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids', None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                targets = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                logits = outputs["logits"]
                
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                
                # Convert logits to probabilities
                probs = torch.sigmoid(logits)
                
                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                
        avg_loss = total_loss / len(self.dataloader)
        
        all_targets = np.vstack(all_targets)
        all_probs = np.vstack(all_probs)
        
        metrics = compute_metrics(all_targets, all_probs, threshold=threshold)
        
        return avg_loss, metrics
