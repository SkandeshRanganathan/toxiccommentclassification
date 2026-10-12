import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger(__name__)

def verify_dataset(data_dir: str):
    """
    Verifies that the Jigsaw toxic comment classification dataset is present.
    Since Kaggle datasets require authentication, we assume the user has manually
    downloaded 'train.csv', 'test.csv', and 'test_labels.csv' to the data_dir.
    """
    required_files = ['train.csv']
    
    os.makedirs(data_dir, exist_ok=True)
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            missing_files.append(file)
            
    if missing_files:
        logger.error(f"Missing required dataset files in {data_dir}: {missing_files}")
        logger.info("Please download the 'Jigsaw Toxic Comment Classification Challenge' dataset from Kaggle:")
        logger.info("https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data")
        logger.info(f"Extract the files and place them in the '{data_dir}' directory.")
        raise FileNotFoundError(f"Missing files: {missing_files}")
    else:
        logger.info(f"Dataset verification successful. Found all required files in {data_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify dataset existence")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory where raw data is stored")
    args = parser.parse_args()
    
    verify_dataset(args.data_dir)
