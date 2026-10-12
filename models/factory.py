"""Single construction point guaranteeing consistent ablation wiring."""

from __future__ import annotations

from typing import Any

from models.baseline_classifier import BaselineClassifier
from models.proposed_model import PrototypeAugmentedClassifier
from models.prototype_memory import PrototypeMemory
from retrieval.prototype_store import PrototypeStore


def build_model(config: Any) -> BaselineClassifier | PrototypeAugmentedClassifier:
    model = config.model
    if not any((model.use_prototypes, model.use_label_correlation)):
        return BaselineClassifier(model.encoder_name, model.num_labels, model.dropout, model.projection_dim)
    memory = None
    if model.use_prototypes:
        store = PrototypeStore.load(config.retrieval.memory_dir)
        memory = PrototypeMemory(store, config.retrieval.top_k, config.retrieval.exclude_self)
    return PrototypeAugmentedClassifier(
        encoder_name=model.encoder_name,
        num_labels=model.num_labels,
        dropout=model.dropout,
        projection_dim=model.projection_dim,
        memory=memory,
        use_prototypes=model.use_prototypes,
        use_adaptive_selector=model.use_adaptive_selector,
        use_prototype_attention=model.use_prototype_attention,
        use_label_correlation=model.use_label_correlation,
        selector_hidden_dim=config.selector.hidden_dim,
        selector_dropout=config.selector.dropout,
        selector_temperature=config.selector.temperature,
        selected_k=config.retrieval.selected_k,
        attention_heads=config.attention.num_heads,
        attention_dropout=config.attention.dropout,
        correlation_hidden_dim=config.label_correlation.hidden_dim,
        correlation_dropout=config.label_correlation.dropout,
    )

