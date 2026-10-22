from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_report import neural_result_summary


class ReportSummaryTests(unittest.TestCase):
    def test_summary_follows_winner_and_best_epoch_after_rerun(self):
        comparison = pd.DataFrame({'model': ['TF-IDF', 'BiGRU', 'PhoBERT'],
                                   'macro_f1': [.2, .6, .4]})
        run = dict(history=[{}, {}, {}, {}], best_epoch=2, train_seconds=60, device='fixture')
        text = neural_result_summary(comparison, run, run)
        self.assertIn('checkpoint tốt nhất epoch 2', text)
        self.assertIn('ngưỡng 0,5: BiGRU.', text)
        self.assertNotIn('Cả hai chọn epoch cuối', text)

    def test_summary_names_tied_winners(self):
        comparison = pd.DataFrame({'model': ['BiGRU', 'PhoBERT'], 'macro_f1': [.6, .6]})
        run = dict(history=[{}], best_epoch=1, train_seconds=1, device='fixture')
        text = neural_result_summary(comparison, run, run)
        self.assertIn('ngưỡng 0,5: BiGRU, PhoBERT.', text)
