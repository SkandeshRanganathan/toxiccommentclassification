import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class AdaptiveSelector(nn.Module):
    """
    Differentiable scoring network to evaluate and select the most relevant retrieved prototypes.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super(AdaptiveSelector, self).__init__()
        
        # We concatenate the query and the prototype embedding to score their relevance
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, query: torch.Tensor, prototypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, embedding_dim)
            prototypes: (batch_size, k, embedding_dim)
            
        Returns:
            selected_prototypes: (batch_size, k, embedding_dim) - weighted by selection scores
            selection_scores: (batch_size, k) - normalized probabilities
        """
        batch_size, k, emb_dim = prototypes.shape
        
        # Expand query to match prototypes
        query_expanded = query.unsqueeze(1).expand(-1, k, -1) # (batch_size, k, embedding_dim)
        
        # Concatenate query and prototypes
        combined = torch.cat([query_expanded, prototypes], dim=-1) # (batch_size, k, embedding_dim * 2)
        
        # Calculate raw scores
        raw_scores = self.scorer(combined).squeeze(-1) # (batch_size, k)
        
        # Normalize scores using Softmax or Sigmoid
        # Softmax creates competition among prototypes
        selection_scores = F.softmax(raw_scores, dim=-1) # (batch_size, k)
        
        # Modulate prototypes by their selection score
        # Even though all k are kept, uninformative ones are scaled down close to zero
        selected_prototypes = prototypes * selection_scores.unsqueeze(-1)
        
        return selected_prototypes, selection_scores
