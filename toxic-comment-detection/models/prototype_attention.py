import torch
import torch.nn as nn
import math
from typing import Tuple

class PrototypeCrossAttention(nn.Module):
    """
    Multi-head Cross Attention where Query is the comment embedding,
    and Keys/Values are the selected prototype embeddings.
    """
    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(PrototypeCrossAttention, self).__init__()
        
        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, prototypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, embedding_dim)
            prototypes: (batch_size, k, embedding_dim)
            
        Returns:
            attn_output: (batch_size, embedding_dim)
            attn_weights: (batch_size, num_heads, 1, k)
        """
        batch_size, k, _ = prototypes.shape
        
        # We treat query as sequence length 1
        q = query.unsqueeze(1) # (batch_size, 1, embedding_dim)
        
        # Linear projections
        q = self.q_proj(q).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = self.k_proj(prototypes).view(batch_size, k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(prototypes).view(batch_size, k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k_proj.transpose(-2, -1)) / self.scale # (batch_size, num_heads, 1, k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)
        
        # Context vector
        context = torch.matmul(attn_weights_dropped, v) # (batch_size, num_heads, 1, head_dim)
        
        # Reshape and project out
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.num_heads * self.head_dim)
        attn_output = self.out_proj(context).squeeze(1) # (batch_size, embedding_dim)
        
        return attn_output, attn_weights
