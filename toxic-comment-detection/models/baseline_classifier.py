import torch
import torch.nn as nn
from models.deberta_encoder import DebertaEncoder

class BaselineClassifier(nn.Module):
    """
    Baseline model for Multi-Label Toxic Comment Classification.
    Architecture: Input -> DeBERTa -> CLS Embedding -> Dropout -> Linear Head
    """
    def __init__(self, encoder_name: str = "microsoft/deberta-v3-base", 
                 embedding_dim: int = 768, 
                 num_labels: int = 6, 
                 dropout_rate: float = 0.1):
        super(BaselineClassifier, self).__init__()
        
        self.encoder = DebertaEncoder(model_name=encoder_name, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(embedding_dim, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None):
        """
        Returns both logits and cls_embeddings since embeddings will be reused later.
        """
        # 1. Get CLS embeddings
        cls_embeddings = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        
        # 2. Dropout and force FP32 (fixes CPU mat1 Half vs Float error)
        out = self.dropout(cls_embeddings).float()
        
        # 3. Linear Classification Head
        logits = self.classifier(out)
        
        return {
            "logits": logits,
            "cls_embeddings": cls_embeddings
        }
