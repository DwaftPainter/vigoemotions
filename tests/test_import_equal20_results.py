"""Reject changes that could silently mix studies or misstate their budget."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import LABELS
from import_equal20_results import digest, file_variant, validate_history, validate_predictions


class Equal20ImportTests(unittest.TestCase):
    def test_line_ending_proof_accepts_crlf_but_rejects_content_change(self):
        local = 'id,text,labels\na,synthetic,[2]\n'.encode()
        self.assertEqual(file_variant(local, digest(local.replace(b'\n', b'\r\n'))), 'CRLF')
        with self.assertRaises(ValueError):
            file_variant(local, digest(local.replace(b'synthetic', b'changed')))

    def test_28_checkpoint_uses_emotion27_and_earliest_tie(self):
        history = [dict(epoch=i, macro_f1=.9 if i == 20 else .1,
                        selection_macro_f1=.6 if i in (14, 15) else .2) for i in range(1, 21)]
        run = dict(history=history, best_epoch=14)
        validate_history(run, 28, .6)
        for best in (15, 20):
            with self.assertRaises(ValueError):
                validate_history(dict(history=history, best_epoch=best), 28, .6)

    def test_partial_history_and_foreign_probabilities_are_rejected(self):
        history = [dict(epoch=i, macro_f1=i/100) for i in range(1, 21)]
        with self.assertRaises(ValueError):
            validate_history(dict(history=history[:12], best_epoch=12), 27, .12)
        with self.assertRaises(ValueError):
            validate_history(dict(history=history, best_epoch=20), 27, .9)

    def test_validation_identity_and_label_order_cannot_be_swapped(self):
        frame = pd.DataFrame(dict(id=['synthetic-a', 'synthetic-b'], label_ids=[[2], [27]]))
        y = np.zeros((2, 27), dtype=int); y[0, 2] = 1
        saved = dict(sample_id=np.array(frame.id), labels=np.array(LABELS),
                     true_label_vector=y, probability_vector=y*.8+.1)
        validate_predictions(saved, frame, 27)
        for key in ('sample_id', 'labels', 'true_label_vector'):
            broken = deepcopy(saved); broken[key] = broken[key][::-1]
            with self.assertRaises(AssertionError):
                validate_predictions(broken, frame, 27)
        broken = deepcopy(saved); broken['probability_vector'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_predictions(broken, frame, 27)
        broken = deepcopy(saved); broken['text'] = np.array(['synthetic', 'synthetic'])
        with self.assertRaises(ValueError):
            validate_predictions(broken, frame, 27)

    def test_default_report_builder_never_falls_back_to_legacy_scores(self):
        import build_report
        with patch('build_equal20_report.main') as current, patch.object(build_report, 'legacy_main') as legacy:
            build_report.main()
            current.assert_called_once_with()
            legacy.assert_not_called()


if __name__ == '__main__':
    unittest.main()
