import logging
import os
import sys
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger that logs to both console and optionally a file.
    
    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if the logger is already set up
    if logger.handlers:
        return logger

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
