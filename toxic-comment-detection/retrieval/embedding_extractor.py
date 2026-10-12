import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import faiss

from retrieval.faiss_index import FaissIndex
from utils.logger import setup_logger

logger = setup_logger(__name__)

def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

def extract_and_build_index(model: torch.nn.Module, dataloader: DataLoader, config: dict, device: torch.device):
    """
    Extracts CLS embeddings from the training set and builds a FAISS index.
    """
    model.eval()
    
    embedding_dim = config["model"]["embedding_dim"]
    index_type = config["model"]["retrieval"]["index_type"]
    do_normalize = config["model"]["retrieval"]["normalize_embeddings"]
    
    index = FaissIndex(dim=embedding_dim, index_type=index_type, use_gpu=torch.cuda.is_available())
    
    all_embeddings = []
    
    logger.info("Extracting embeddings for Prototype Memory...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
                
            labels = batch['labels'].numpy()
            texts = batch['text']
            
            outputs = model(input_ids, attention_mask, token_type_ids)
            cls_embeddings = outputs["cls_embeddings"]
            
            if do_normalize:
                cls_embeddings = normalize_embeddings(cls_embeddings)
                
            cls_embeddings = cls_embeddings.cpu().numpy()
            
            index.add(cls_embeddings, labels, texts)
            all_embeddings.append(cls_embeddings)
            
    # Save the original embeddings as well for the attention mechanism
    all_embeddings = np.vstack(all_embeddings)
    memory_dir = os.path.join(config["project"]["output_dir"], "prototype_memory")
    os.makedirs(memory_dir, exist_ok=True)
    
    np.save(os.path.join(memory_dir, "all_embeddings.npy"), all_embeddings)
    index.save(memory_dir)
    
    return index, all_embeddings
