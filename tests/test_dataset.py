from __future__ import annotations

import unittest

import pandas as pd

from data.dataset import ToxicCommentDataset
from utils.constants import LABELS


class FakeTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        del text, kwargs
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}


class DatasetTests(unittest.TestCase):
    def test_dynamic_tokenization_and_multilabel_tensor(self) -> None:
        frame = pd.DataFrame([{"id": "x", "comment_text": "sample", **{label: int(i % 2) for i, label in enumerate(LABELS)}}])
        item = ToxicCommentDataset(frame, FakeTokenizer(), LABELS, 32)[0]
        self.assertEqual(tuple(item["labels"].shape), (6,))
        self.assertEqual(item["sample_id"], "x")


if __name__ == "__main__":
    unittest.main()

