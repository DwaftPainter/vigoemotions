"""Protect evaluation scope, displayed metrics and slide geometry."""
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from academic_slide_design import NAMES, slide_definitions
from academic_slides import H, W, SlideCanvas, render_deck
from import_equal20_results import OUT


class AcademicPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slides = slide_definitions()

    def test_performance_table_preserves_both_full_label_evaluations(self):
        comparison = pd.read_csv(OUT / 'model_comparison.csv')
        rows = self.slides[5]['table']['rows']
        self.assertEqual(len(rows), 6)
        for row in rows:
            key = next(key for key, name in NAMES.items() if name == row[1])
            run = comparison[(comparison.label_count == int(row[0])) & (comparison.model == key)].iloc[0]
            self.assertEqual(row[2:6], [f'{run[m]:.4f}' for m in
                                      ('macro_f1', 'micro_f1', 'hamming_loss', 'lrap')])
            if key != 'tfidf_lr':
                self.assertEqual(run.epochs_trained, 20)
                self.assertEqual(row[6], str(int(run.best_epoch)))

    def test_neutral_effect_uses_shared27_predictions_and_neutral_counts(self):
        paired = self.slides[6]['table']['rows']
        neutral = self.slides[7]['table']['rows']
        for i, key in enumerate(NAMES):
            with np.load(OUT / f'evidence/27/{key}/val_predictions.npz', allow_pickle=False) as a:
                full27 = f1_score(a['true_label_vector'], a['probability_vector'] >= .5,
                                 average='macro', zero_division=0)
            with np.load(OUT / f'evidence/28/{key}/val_predictions.npz', allow_pickle=False) as a:
                truth, predicted = a['true_label_vector'], a['probability_vector'] >= .5
                shared27 = f1_score(truth[:, :27], predicted[:, :27], average='macro', zero_division=0)
                counts = [np.count_nonzero(truth[:, 27] & predicted[:, 27]),
                          np.count_nonzero(~truth[:, 27].astype(bool) & predicted[:, 27]),
                          np.count_nonzero(truth[:, 27] & ~predicted[:, 27])]
            self.assertEqual(paired[i][1:], [f'{full27:.4f}', f'{shared27:.4f}',
                                            f'{100 * (shared27 - full27):+.2f}'])
            self.assertEqual(neutral[i][-3:], [str(n) for n in counts])

    def test_all_slides_render_without_text_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'slides.pdf'
            boxes = render_deck(self.slides, target)
            pages = PdfReader(target).pages
            self.assertEqual(len(pages), 15)
            self.assertEqual(sum('table' in slide for slide in self.slides), 7)
            for page in pages:
                self.assertEqual((float(page.mediabox.width), float(page.mediabox.height)), (W, H))
            for a, b in combinations(boxes, 2):
                if a['slide'] != b['slide']:
                    continue
                dx = min(a['x'] + a['width'], b['x'] + b['width']) - max(a['x'], b['x'])
                dy = min(a['y'] + a['height'], b['y'] + b['height']) - max(a['y'], b['y'])
                self.assertFalse(dx > 2 and dy > 2, f'Text collision: {a} / {b}')

    def test_renderer_rejects_clipped_text(self):
        with tempfile.TemporaryDirectory() as directory:
            canvas = SlideCanvas(Path(directory) / 'overflow.pdf', 1)
            with self.assertRaisesRegex(ValueError, 'text overflow'):
                canvas.text('A title that must wrap across several lines', 64, 300, 80,
                            size=30, height=30)
            with self.assertRaisesRegex(ValueError, 'outside page'):
                canvas.text('Outside', W - 10, 300, 80)
            canvas.c.save()


if __name__ == '__main__':
    unittest.main()
