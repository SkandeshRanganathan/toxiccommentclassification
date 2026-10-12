import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

from utils.logger import setup_logger
from utils.seed import set_seed

logger = setup_logger(__name__)

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_data(config):
    """
    Loads raw data, cleans it, splits into train/val/test, and saves processed files.
    """
    raw_path = config["data"]["raw_data_path"]
    processed_dir = config["data"]["processed_dir"]
    text_col = config["data"]["text_col"]
    label_cols = config["data"]["label_cols"]
    seed = config["project"]["seed"]
    
    os.makedirs(processed_dir, exist_ok=True)
    
    logger.info(f"Loading raw dataset from {raw_path}")
    df = pd.read_csv(raw_path)
    
    # Basic cleaning
    initial_len = len(df)
    df.dropna(subset=[text_col], inplace=True)
    
    # Strip all newlines from comments to prevent Pandas C-engine OOM crashes later!
    df[text_col] = df[text_col].astype(str).str.replace(r'\r|\n', ' ', regex=True)
    
    logger.info(f"Removed {initial_len - len(df)} rows with missing text.")
    
    # Ensure label columns exist and are numeric
    for col in label_cols:
        if col not in df.columns:
            raise ValueError(f"Label column '{col}' not found in dataset.")
        df[col] = df[col].astype(int)
        
    # Splitting into train, val, test
    # Default split: 80% train, 10% val, 10% test
    train_split = config["data"].get("train_split", 0.8)
    val_split = config["data"].get("val_split", 0.1)
    # The remainder is test
    
    logger.info(f"Splitting dataset with ratio Train: {train_split}, Val: {val_split}, Test: {1 - train_split - val_split:.2f}")
    
    train_df, temp_df = train_test_split(df, test_size=(1 - train_split), random_state=seed)
    
    relative_val_split = val_split / (1 - train_split)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - relative_val_split), random_state=seed)
    
    # Save splits
    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved Train set: {len(train_df)} samples to {train_path}")
    logger.info(f"Saved Val set: {len(val_df)} samples to {val_path}")
    logger.info(f"Saved Test set: {len(test_df)} samples to {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    preprocess_data(config)
