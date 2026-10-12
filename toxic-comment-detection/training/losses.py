import torch
import torch.nn as nn

class MultiLabelLoss(nn.Module):
    """
    Standard Binary Cross Entropy with Logits for Multi-Label Classification.
    """
    def __init__(self, pos_weight=None):
        super(MultiLabelLoss, self).__init__()
        # pos_weight can be used to handle class imbalance
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized logits from the model (batch_size, num_classes)
            targets: Binary targets (batch_size, num_classes)
        """
        return self.loss_fn(logits, targets)

# Additional auxiliary losses (e.g., prototype contrastive loss) can be added here in later phases.
