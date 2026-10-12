import torch
import torch.nn as nn

class LabelCorrelation(nn.Module):
    """
    Models dependencies among the toxic labels (e.g., Obscene and Insult often co-occur).
    Uses a learnable label embedding matrix and self-attention across the labels.
    """
    def __init__(self, num_labels: int, embedding_dim: int):
        super(LabelCorrelation, self).__init__()
        
        self.num_labels = num_labels
        
        # Learnable embeddings for each label
        self.label_embeddings = nn.Parameter(torch.randn(1, num_labels, embedding_dim))
        
        # Self-attention to model correlations between labels
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Combines the input features with the label correlation matrix.
        
        Args:
            features: (batch_size, embedding_dim)
            
        Returns:
            label_aware_features: (batch_size, num_labels, embedding_dim)
        """
        batch_size = features.shape[0]
        
        # Expand label embeddings for the batch
        labels_emb = self.label_embeddings.expand(batch_size, -1, -1) # (batch_size, num_labels, embedding_dim)
        
        # Add the comment feature to the label embeddings as context
        # (batch_size, num_labels, embedding_dim)
        contextualized_labels = labels_emb + features.unsqueeze(1)
        
        # Self-attention across labels
        attn_out, _ = self.attention(contextualized_labels, contextualized_labels, contextualized_labels)
        
        out1 = self.norm1(contextualized_labels + attn_out)
        ffn_out = self.ffn(out1)
        label_aware_features = self.norm2(out1 + ffn_out)
        
        return label_aware_features
