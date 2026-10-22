import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import load_data, multihot, parse_labels, scores, style_masks


class DataContractTests(unittest.TestCase):
    def test_teencode_uses_unicode_word_boundaries(self):
        texts = ['không', 'kệ', 'ko vui', 'đc rồi', 'vui', 'KO!']
        for dtype in ('object', 'str'):
            df = pd.DataFrame({'text': pd.Series(texts, dtype=dtype),
                               'normalized': pd.Series(texts, dtype=dtype)})
            self.assertEqual(style_masks(df)['teencode'].tolist(),
                             [False, False, True, True, False, True])

    def test_neutral_mixed_and_duplicate_labels(self):
        self.assertEqual(multihot([27]).sum(), 0)
        self.assertEqual(multihot([27, 2, 2]).sum(), 1)
        self.assertEqual(multihot([27, 2])[2], 1)

    def test_invalid_labels(self):
        for labels in ([], [28], [-1], [True], ['2'], 'not a list'):
            with self.assertRaises((ValueError, SyntaxError)):
                parse_labels(labels)

    def test_metrics_known_example(self):
        y = np.array([[1, 0], [0, 1]])
        p = np.array([[.9, .2], [.8, .7]])
        m = scores(y, p)
        self.assertAlmostEqual(m['micro_f1'], .8)
        self.assertAlmostEqual(m['macro_f1'], 5/6)
        self.assertAlmostEqual(m['hamming_loss'], .25)
        self.assertAlmostEqual(m['lrap'], .75)

    def test_lrap_matches_float64_reference_for_float32_predictions(self):
        from sklearn.metrics import label_ranking_average_precision_score
        rng = np.random.default_rng(42)
        y = (rng.random((2066, 27)) < .07).astype(int)
        p = rng.random((2066, 27)).astype(np.float32)
        reference = label_ranking_average_precision_score(y, p.astype(np.float64))
        self.assertAlmostEqual(scores(y, p)['lrap'], reference, places=12)

    def test_training_has_no_heldout_text(self):
        frames, excluded = load_data()
        self.assertEqual(sum(map(len, frames.values())) + len(excluded), 20664)
        for s in ('val', 'test'):
            self.assertFalse(set(frames['train'].normalized) & set(frames[s].normalized))


if __name__ == '__main__':
    unittest.main()
