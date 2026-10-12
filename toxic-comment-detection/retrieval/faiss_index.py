import faiss
import numpy as np
import torch
import os
from typing import Tuple, List

from utils.logger import setup_logger

logger = setup_logger(__name__)

class FaissIndex:
    """
    Wrapper for FAISS Index.
    Supports Inner Product (Cosine Similarity if normalized) and L2.
    """
    def __init__(self, dim: int, index_type: str = "l2", use_gpu: bool = False):
        self.dim = dim
        self.index_type = index_type
        
        if index_type == "l2":
            self.index = faiss.IndexFlatL2(dim)
        elif index_type == "inner_product":
            self.index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            logger.info("Using GPU for FAISS index.")
            
        # Parallel arrays to store metadata (labels, texts)
        self.labels = []
        self.texts = []
        
    def add(self, embeddings: np.ndarray, labels: np.ndarray, texts: List[str]):
        """
        Add embeddings and metadata to the index.
        """
        assert embeddings.shape[1] == self.dim
        self.index.add(embeddings.astype('float32'))
        self.labels.extend(labels)
        self.texts.extend(texts)
        
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[str]]]:
        """
        Search the index for top-k nearest neighbors.
        Returns: distances, indices, retrieved_labels, retrieved_texts
        """
        assert queries.shape[1] == self.dim
        distances, indices = self.index.search(queries.astype('float32'), k)
        
        # Gather metadata
        retrieved_labels = np.zeros((queries.shape[0], k, len(self.labels[0])), dtype=np.float32)
        retrieved_texts = []
        
        for i in range(queries.shape[0]):
            batch_texts = []
            for j in range(k):
                idx = indices[i, j]
                if idx != -1: # FAISS returns -1 if there are not enough neighbors
                    retrieved_labels[i, j] = self.labels[idx]
                    batch_texts.append(self.texts[idx])
                else:
                    batch_texts.append("")
            retrieved_texts.append(batch_texts)
            
        return distances, indices, retrieved_labels, retrieved_texts
        
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "faiss.index")
        
        if self.use_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        else:
            index_to_save = self.index
            
        faiss.write_index(index_to_save, index_path)
        np.save(os.path.join(directory, "labels.npy"), np.array(self.labels))
        
        # Save texts
        import json
        with open(os.path.join(directory, "texts.json"), "w") as f:
            json.dump(self.texts, f)
            
        logger.info(f"FAISS index and metadata saved to {directory}")
        
    def load(self, directory: str):
        index_path = os.path.join(directory, "faiss.index")
        
        self.index = faiss.read_index(index_path)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
        self.labels = np.load(os.path.join(directory, "labels.npy")).tolist()
        
        import json
        with open(os.path.join(directory, "texts.json"), "r") as f:
            self.texts = json.load(f)
            
        logger.info(f"FAISS index and metadata loaded from {directory}")
