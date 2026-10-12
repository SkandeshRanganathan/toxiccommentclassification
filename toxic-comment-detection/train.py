import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import os

from utils.logger import setup_logger
from utils.seed import set_seed
from data.dataset import ToxicCommentDataset
from models.baseline_classifier import BaselineClassifier
from models.full_model import FullToxicModel
from training.losses import MultiLabelLoss
from training.trainer import Trainer
from retrieval.embedding_extractor import extract_and_build_index

logger = setup_logger(__name__)

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--build_memory", action="store_true", help="Build FAISS memory from training data")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Datasets & DataLoaders
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder_name"])
    
    processed_dir = config["data"]["processed_dir"]
    train_dataset = ToxicCommentDataset(
        data_path=os.path.join(processed_dir, "train.csv"),
        tokenizer=tokenizer,
        text_col=config["data"]["text_col"],
        label_cols=config["data"]["label_cols"],
        max_seq_length=config["data"]["max_seq_length"]
    )
    train_dataset.df = train_dataset.df.head(32)
    val_dataset = ToxicCommentDataset(
        data_path=os.path.join(processed_dir, "val.csv"),
        tokenizer=tokenizer,
        text_col=config["data"]["text_col"],
        label_cols=config["data"]["label_cols"],
        max_seq_length=config["data"]["max_seq_length"]
    )
    val_dataset.df = val_dataset.df.head(8)
    
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["eval_batch_size"], shuffle=False, num_workers=0)
    
    # 2. Build Memory if requested
    if args.build_memory:
        logger.info("Building FAISS Prototype Memory using Baseline Encoder...")
        baseline = BaselineClassifier(
            encoder_name=config["model"]["encoder_name"],
            embedding_dim=config["model"]["embedding_dim"],
            num_labels=config["model"]["num_labels"]
        ).to(device)
        
        # Ideally, you'd load a pre-trained baseline here before extracting.
        # Assuming the baseline is at least somewhat trained, or we just want zero-shot embeddings.
        extract_and_build_index(baseline, train_loader, config, device)
        logger.info("Memory built successfully. Please restart training without --build_memory to train full model.")
        return

    # 3. Model
    if config["model"]["use_faiss_memory"]:
        model = FullToxicModel(config, device).to(device)
    else:
        model = BaselineClassifier(
            encoder_name=config["model"]["encoder_name"],
            embedding_dim=config["model"]["embedding_dim"],
            num_labels=config["model"]["num_labels"]
        ).to(device)
        
    model = model.float() # Force all weights to float32 to prevent NaN gradients and precision mismatches
        
    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config["training"]["learning_rate"]), 
        weight_decay=config["training"]["weight_decay"]
    )
    criterion = MultiLabelLoss()
    
    # 5. Train
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, config)
    trainer.train()
    
if __name__ == "__main__":
    main()
