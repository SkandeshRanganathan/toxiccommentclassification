import argparse
import yaml
import torch
from transformers import AutoTokenizer
import os
import faiss

from models.full_model import FullToxicModel
from utils.checkpoint import load_checkpoint
from utils.visualization import plot_attention_heatmap
from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def infer(text: str, model: torch.nn.Module, tokenizer, config: dict, device: torch.device):
    model.eval()
    
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=config["data"]["max_seq_length"],
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
        
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
        
    logits = outputs["logits"]
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    
    labels = config["data"]["label_cols"]
    default_threshold = config["evaluation"]["default_threshold"]
    
    print("\n" + "="*50)
    print(f"Input Text: {text}")
    print("="*50)
    
    print("\nPredictions:")
    for i, label in enumerate(labels):
        prob = probs[i]
        is_toxic = prob >= default_threshold
        marker = "[X]" if is_toxic else "[ ]"
        print(f"{marker} {label.ljust(15)}: {prob:.4f}")
        
    # If FAISS was used, show prototypes
    if config["model"]["use_faiss_memory"]:
        print("\nRetrieved Prototypes:")
        # We need to manually query the index to get the raw text since the model only returns tensors
        cls_embs = outputs["cls_embeddings"]
        if config["model"]["retrieval"]["normalize_embeddings"]:
            cls_embs = torch.nn.functional.normalize(cls_embs, p=2, dim=1)
            
        cls_embs_np = cls_embs.cpu().numpy()
        distances, indices, ret_labels, ret_texts = model.memory.faiss_index.search(cls_embs_np, config["model"]["retrieval"]["k_nearest"])
        
        for j in range(len(ret_texts[0])):
            print(f"\nPrototype {j+1} (Dist: {distances[0][j]:.4f})")
            print(f"Text: {ret_texts[0][j][:100]}...")
            
            # Print label of prototype
            p_labels = ret_labels[0][j]
            active_labels = [labels[i] for i, val in enumerate(p_labels) if val == 1]
            print(f"Labels: {active_labels}")
            
        # Attention
        if outputs.get("attn_weights") is not None:
            attn_weights = outputs["attn_weights"].cpu().numpy()
            out_path = os.path.join(config["project"]["output_dir"], "inference_attention.png")
            plot_attention_heatmap(attn_weights, text, ret_texts[0], out_path)
            print(f"\nSaved attention heatmap to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder_name"])
    
    # Load Model
    if config["model"]["use_faiss_memory"]:
        model = FullToxicModel(config, device).to(device)
    else:
        from models.baseline_classifier import BaselineClassifier
        model = BaselineClassifier(
            encoder_name=config["model"]["encoder_name"],
            embedding_dim=config["model"]["embedding_dim"],
            num_labels=config["model"]["num_labels"]
        ).to(device)
        
    model = model.float() # Force float32 to prevent issues
        
    load_checkpoint(args.checkpoint, model)
    
    infer(args.text, model, tokenizer, config, device)

if __name__ == "__main__":
    main()
