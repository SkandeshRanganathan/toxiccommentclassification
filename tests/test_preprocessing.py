from __future__ import annotations

import unittest

import pandas as pd

from data.preprocess import validate_and_clean
from utils.constants import LABELS


class PreprocessingTests(unittest.TestCase):
    def test_missing_rows_are_removed_and_labels_preserved(self) -> None:
        rows = [{"id": "a", "comment_text": "hello", **{label: 0 for label in LABELS}}, {"id": "b", "comment_text": None, **{label: 1 for label in LABELS}}]
        clean = validate_and_clean(pd.DataFrame(rows), "id", "comment_text", LABELS)
        self.assertEqual(len(clean), 1)
        self.assertEqual(list(clean.columns), ["id", "comment_text", *LABELS])

    def test_nonbinary_label_is_rejected(self) -> None:
        row = {"id": "a", "comment_text": "hello", **{label: 0 for label in LABELS}}
        row["toxic"] = 2
        with self.assertRaises(ValueError):
            validate_and_clean(pd.DataFrame([row]), "id", "comment_text", LABELS)


if __name__ == "__main__":
    unittest.main()

