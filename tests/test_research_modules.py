from __future__ import annotations

import unittest

import numpy as np
import torch

from models.adaptive_selector import AdaptivePrototypeSelector
from models.label_correlation import LabelCorrelationModel
from models.prototype_attention import PrototypeAwareCrossAttention
from models.threshold_optimizer import PerLabelThresholdOptimizer
from utils.metrics import multilabel_metrics


class ResearchModuleTests(unittest.TestCase):
    def test_selector_is_differentiable_and_normalized(self) -> None:
        selector = AdaptivePrototypeSelector(8, 16, hard_top_k=2)
        query = torch.randn(3, 8, requires_grad=True)
        prototypes = torch.randn(3, 4, 8)
        output = selector(query, prototypes, torch.randn(3, 4), torch.randint(0, 2, (3, 4, 6)).float(), torch.rand(3, 6), torch.ones(3, 4, dtype=torch.bool))
        self.assertTrue(torch.allclose(output.weights.sum(1), torch.ones(3)))
        output.weights[:, 0].sum().backward()
        self.assertIsNotNone(query.grad)

    def test_attention_and_correlation_shapes(self) -> None:
        query, prototypes = torch.randn(2, 8), torch.randn(2, 3, 8)
        attention = PrototypeAwareCrossAttention(8, 2)(query, prototypes)
        self.assertEqual(tuple(attention.fused.shape), (2, 8))
        correlated = LabelCorrelationModel(8, 6, 12)(attention.fused)
        self.assertEqual(tuple(correlated.logits.shape), (2, 6))
        self.assertEqual(tuple(correlated.correlation_matrix.shape), (6, 6))

    def test_threshold_optimizer_improves_nonstandard_boundary(self) -> None:
        targets = np.array([[0], [0], [1], [1]])
        probabilities = np.array([[0.05], [0.10], [0.25], [0.35]])
        result = PerLabelThresholdOptimizer().fit(probabilities, targets)
        self.assertLess(result.thresholds[0], 0.5)
        self.assertEqual(result.per_label_f1[0], 1.0)

    def test_exact_match_alias(self) -> None:
        targets = np.array([[1, 0], [0, 1]])
        metrics = multilabel_metrics(targets, targets.astype(float), label_names=["a", "b"])
        self.assertEqual(metrics["exact_match_ratio"], metrics["subset_accuracy"])


if __name__ == "__main__":
    unittest.main()

