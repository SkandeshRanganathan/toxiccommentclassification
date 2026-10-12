import os
import torch
import torch.nn as nn
from typing import Dict, Any

from utils.logger import setup_logger

logger = setup_logger(__name__)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, 
                    epoch: int, val_loss: float, config: Dict[str, Any], is_best: bool = False):
    """
    Saves a model checkpoint.
    """
    checkpoint_dir = config["project"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss': val_loss
    }
    
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
        
    filepath = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(state, filepath)
    logger.info(f"Saved checkpoint to {filepath}")
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(state, best_filepath)
        logger.info(f"Saved best model to {best_filepath}")

def load_checkpoint(filepath: str, model: nn.Module, optimizer: torch.optim.Optimizer = None, scheduler: Any = None):
    """
    Loads a model checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
        
    logger.info(f"Loading checkpoint from {filepath}")
    state = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(state['state_dict'])
    
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
        
    if scheduler is not None and 'scheduler' in state:
        scheduler.load_state_dict(state['scheduler'])
        
    return state.get('epoch', 0), state.get('val_loss', float('inf'))
