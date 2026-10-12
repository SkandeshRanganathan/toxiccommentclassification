import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class DebertaEncoder(nn.Module):
    """
    DeBERTa-v3 Encoder that extracts the [CLS] embedding for a given sequence.
    """
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", dropout_rate: float = 0.1):
        super(DebertaEncoder, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = dropout_rate
        self.config.attention_probs_dropout_prob = dropout_rate
        
        # We use AutoModel to extract embeddings rather than AutoModelForSequenceClassification
        self.deberta = AutoModel.from_pretrained(model_name, config=self.config)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_length)
            attention_mask: Tensor of shape (batch_size, seq_length)
            token_type_ids: Tensor of shape (batch_size, seq_length), optional
            
        Returns:
            cls_embedding: Tensor of shape (batch_size, hidden_size)
        """
        # DeBERTa forward pass
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # DeBERTa-v3 does not always return pooler_output by default
        # We take the hidden state of the first token [CLS]
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        
        return cls_embedding
