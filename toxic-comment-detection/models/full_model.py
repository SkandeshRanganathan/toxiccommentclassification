import torch
import torch.nn as nn

from models.deberta_encoder import DebertaEncoder
from models.prototype_memory import PrototypeMemory
from models.adaptive_selector import AdaptiveSelector
from models.prototype_attention import PrototypeCrossAttention
from models.label_correlation import LabelCorrelation

class FullToxicModel(nn.Module):
    """
    Complete Proposed Model:
    DeBERTa -> Memory Retrieval -> Adaptive Selection -> Cross Attention -> Fusion -> Label Correlation
    """
    def __init__(self, config: dict, device: torch.device):
        super(FullToxicModel, self).__init__()
        self.config = config
        self.device = device
        
        embedding_dim = config["model"]["embedding_dim"]
        num_labels = config["model"]["num_labels"]
        dropout_rate = config["model"]["dropout_rate"]
        
        self.encoder = DebertaEncoder(
            model_name=config["model"]["encoder_name"], 
            dropout_rate=dropout_rate
        )
        
        # FREEZE ENCODER FOR DEMO: Saves massive RAM and prevents NaN gradients
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.use_faiss = config["model"].get("use_faiss_memory", True)
        self.use_adaptive = config["model"].get("use_adaptive_selection", True)
        self.use_attention = config["model"].get("use_prototype_attention", True)
        self.use_label_corr = config["model"].get("use_label_correlation", True)
        
        # Prototype Components
        if self.use_faiss:
            self.memory = PrototypeMemory(config, device)
            
            if self.use_adaptive:
                self.selector = AdaptiveSelector(embedding_dim)
                
            if self.use_attention:
                num_heads = config["model"]["attention"]["num_heads"]
                self.cross_attention = PrototypeCrossAttention(embedding_dim, num_heads, dropout_rate)
                # Learnable fusion weight
                self.fusion_weight = nn.Parameter(torch.tensor(0.5))
            else:
                # If no attention, simple concatenation fusion
                self.fusion_layer = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Label Correlation
        if self.use_label_corr:
            self.label_correlation = LabelCorrelation(num_labels, embedding_dim)
            # Final classifier from label-aware features to scalars
            self.classifier = nn.Linear(embedding_dim, 1)
        else:
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(embedding_dim, num_labels)
            
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None):
        
        cls_embeddings = self.encoder(input_ids, attention_mask, token_type_ids)
        cls_embeddings = cls_embeddings.float() # Force float32 to fix HuggingFace Deberta returning Half
        
        fusion_out = cls_embeddings
        attn_weights = None
        selection_scores = None
        
        if self.use_faiss:
            # 1. Retrieve
            prototypes, ret_labels, dists = self.memory.retrieve(cls_embeddings)
            
            # 2. Select
            if self.use_adaptive:
                prototypes, selection_scores = self.selector(cls_embeddings, prototypes)
                
            # 3. Attention & Fusion
            if self.use_attention:
                attn_out, attn_weights = self.cross_attention(cls_embeddings, prototypes)
                fusion_out = self.fusion_weight * attn_out + (1 - self.fusion_weight) * cls_embeddings
            else:
                # Average pooling over prototypes and concat
                proto_avg = prototypes.mean(dim=1)
                fusion_out = self.fusion_layer(torch.cat([cls_embeddings, proto_avg], dim=-1))
                
        # 4. Classification
        if self.use_label_corr:
            label_aware_features = self.label_correlation(fusion_out) # (batch, num_labels, emb_dim)
            logits = self.classifier(label_aware_features).squeeze(-1) # (batch, num_labels)
        else:
            out = self.dropout(fusion_out)
            logits = self.classifier(out) # (batch, num_labels)
            
        return {
            "logits": logits,
            "cls_embeddings": cls_embeddings,
            "attn_weights": attn_weights,
            "selection_scores": selection_scores
        }
