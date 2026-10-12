from __future__ import annotations

import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("faiss"), "FAISS is not installed")
class FaissIndexTests(unittest.TestCase):
    def test_cosine_neighbor_order(self) -> None:
        from retrieval.faiss_index import FaissCosineIndex

        index = FaissCosineIndex(2)
        index.add(np.array([[1, 0], [0, 1]], dtype=np.float32))
        scores, rows = index.search(np.array([[0.9, 0.1]], dtype=np.float32), 2)
        self.assertEqual(rows[0, 0], 0)
        self.assertGreater(scores[0, 0], scores[0, 1])


if __name__ == "__main__":
    unittest.main()

