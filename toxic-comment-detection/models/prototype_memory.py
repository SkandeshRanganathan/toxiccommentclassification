import torch
import torch.nn as nn
import numpy as np
import os
from typing import Tuple

class PrototypeMemory(nn.Module):
    """
    PyTorch wrapper for the Prototype Memory.
    During training/inference, we use the pre-built FAISS index to find neighbors.
    We maintain a frozen embedding matrix to map retrieved indices back to PyTorch tensors.
    """
    def __init__(self, config: dict, device: torch.device):
        super(PrototypeMemory, self).__init__()
        self.config = config
        self.device = device
        
        self.k = config["model"]["retrieval"]["k_nearest"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.do_normalize = config["model"]["retrieval"]["normalize_embeddings"]
        
        memory_dir = os.path.join(config["project"]["output_dir"], "prototype_memory")
        
        # Load embeddings into a frozen Embedding layer for gradient flow in attention later
        embeddings_np = np.load(os.path.join(memory_dir, "all_embeddings.npy"))
        self.embeddings = nn.Embedding.from_pretrained(
            torch.from_numpy(embeddings_np).float(), freeze=True
        ).to(device)
        
        # We also need the FAISS index for dynamic retrieval
        # Since FAISS is not differentiable, retrieval is a pre-computation step for the forward pass
        from retrieval.faiss_index import FaissIndex
        index_type = config["model"]["retrieval"]["index_type"]
        self.faiss_index = FaissIndex(dim=self.embedding_dim, index_type=index_type, use_gpu=torch.cuda.is_available())
        self.faiss_index.load(memory_dir)
        
    def retrieve(self, query_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves top-k prototype embeddings and their labels for the given queries.
        
        Args:
            query_embeddings: Tensor of shape (batch_size, embedding_dim)
            
        Returns:
            retrieved_embs: (batch_size, k, embedding_dim)
            retrieved_labels: (batch_size, k, num_labels)
            distances_tensor: (batch_size, k)
        """
        if self.do_normalize:
            query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
            
        queries_np = query_embeddings.detach().cpu().numpy()
        distances, indices, labels, _ = self.faiss_index.search(queries_np, self.k)
        
        # Convert to PyTorch tensors
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        indices_tensor = torch.clamp(indices_tensor, min=0) # Fix FAISS returning -1 when not enough neighbors
        retrieved_labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
        distances_tensor = torch.tensor(distances, dtype=torch.float32, device=self.device)
        
        # Use embedding lookup to get tensors so we can participate in attention operations
        # (Though prototypes themselves are frozen)
        retrieved_embs = self.embeddings(indices_tensor)
        
        return retrieved_embs, retrieved_labels, distances_tensor
