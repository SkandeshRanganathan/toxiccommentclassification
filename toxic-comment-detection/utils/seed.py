import random
import os
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
        
    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
