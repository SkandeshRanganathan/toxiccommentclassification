"""Extract training CLS vectors from a baseline checkpoint and build FAISS memory."""

from __future__ import annotations

import argparse

from models.baseline_classifier import BaselineClassifier
from retrieval.embedding_extractor import extract_embeddings
from retrieval.prototype_store import PrototypeStore
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.device import resolve_device
from utils.runtime import build_loader, build_tokenizer
from utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config.training.seed, config.training.deterministic)
    policy = resolve_device(config.training.mixed_precision)
    tokenizer = build_tokenizer(config)
    loader = build_loader(config, tokenizer, "train", False)
    model = BaselineClassifier(config.model.encoder_name, config.model.num_labels, config.model.dropout, config.model.projection_dim).to(policy.device)
    load_checkpoint(args.checkpoint, model, device=policy.device)
    extracted = extract_embeddings(model, loader, policy.device, policy.autocast)
    store = PrototypeStore.build(extracted.embeddings, extracted.sample_ids, extracted.comments, extracted.labels, config.data.labels, config.retrieval.index_type, config.retrieval.nlist, config.retrieval.nprobe, config.retrieval.hnsw_m)
    destination = args.output or config.retrieval.memory_dir
    store.save(destination)
    print(f"Saved {len(extracted.sample_ids)} prototypes to {destination}")


if __name__ == "__main__":
    main()

