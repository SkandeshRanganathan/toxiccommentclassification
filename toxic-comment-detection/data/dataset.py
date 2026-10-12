import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from typing import List, Dict, Any

class ToxicCommentDataset(Dataset):
    """
    Custom PyTorch Dataset for Toxic Comment Classification.
    Performs on-the-fly tokenization using HuggingFace Tokenizer.
    """
    def __init__(self, data_path: str, tokenizer: Any, text_col: str, label_cols: List[str], max_seq_length: int = 128):
        """
        Args:
            data_path (str): Path to the CSV file containing the data.
            tokenizer (Any): Pre-instantiated HuggingFace Tokenizer.
            text_col (str): Column name for the text.
            label_cols (List[str]): List of column names for the multi-label targets.
            max_seq_length (int): Maximum sequence length for tokenization.
        """
        # The data is now cleaned of newlines during preprocessing, so the fast C engine works perfectly
        self.df = pd.read_csv(data_path)
        self.text_col = text_col
        self.label_cols = label_cols
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        
        # Multi-label targets as float32 for BCEWithLogitsLoss
        labels = torch.tensor([row[col] for col in self.label_cols], dtype=torch.float32)
        
        # Tokenize on the fly
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension added by return_tensors="pt"
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "text": text # Keeping original text for prototype memory and inference visualization
        }
        
        # Check if token_type_ids are returned (DeBERTa v3 might not use them natively, but just in case)
        if "token_type_ids" in encoding:
             item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
             
        return item
