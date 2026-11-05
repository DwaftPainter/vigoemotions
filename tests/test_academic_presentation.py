"""Protect evaluation scope, displayed metrics and slide geometry."""
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.metrics import f1_score, hamming_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from academic_slide_design import NAMES, slide_definitions
from academic_slides import H, W, SlideCanvas, render_deck
from common import LABELS
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
            self.assertEqual(len(pages), 18)
            self.assertEqual(sum('table' in slide for slide in self.slides), 11)
            for page in pages:
                self.assertEqual((float(page.mediabox.width), float(page.mediabox.height)), (W, H))
            for a, b in combinations(boxes, 2):
                if a['slide'] != b['slide']:
                    continue
                dx = min(a['x'] + a['width'], b['x'] + b['width']) - max(a['x'], b['x'])
                dy = min(a['y'] + a['height'], b['y'] + b['height']) - max(a['y'], b['y'])
                self.assertFalse(dx > 2 and dy > 2, f'Text collision: {a} / {b}')

    def test_research_questions_have_dedicated_readable_pages(self):
        for question in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5'):
            dedicated = [s for s in self.slides if s.get('rq') == question]
            self.assertTrue(dedicated, question)
            self.assertTrue(all(s['title'].startswith(question + ' ·') for s in dedicated))
            self.assertTrue(all(s['notes'] and s['takeaway'] for s in dedicated))
        self.assertFalse(any('checklist' in s['title'].lower() for s in self.slides))

    def test_q3_joint_recall_is_recomputed_from_all_three_models(self):
        rows = next(s for s in self.slides if s.get('rq') == 'Q3')['table']['rows']
        pair = [LABELS.index('disappointment'), LABELS.index('sadness')]
        self.assertEqual(len(rows), 3)
        for row, model in zip(rows, NAMES):
            with np.load(OUT / f'evidence/27/{model}/val_predictions.npz', allow_pickle=False) as a:
                positive = a['true_label_vector'][:, pair].all(axis=1)
                both_predicted = (a['probability_vector'][:, pair] >= .5).all(axis=1)
                support = int(positive.sum())
                joint_tp = int((positive & both_predicted).sum())
            self.assertEqual(row, [NAMES[model], str(joint_tp), str(support), f'{joint_tp/support:.2%}'])

    def test_q4_covers_all_groups_with_compatible_counts_and_deltas(self):
        rows = next(s for s in self.slides if s.get('rq') == 'Q4')['table']['rows']
        groups = ('all', 'emoji_heavy_ge_2', 'short_le_5', 'teencode')
        self.assertEqual(len(rows), len(groups))
        for row, group in zip(rows, groups):
            sources = [pd.read_csv(OUT / f'analysis/27_emotion27/{m}/tables/{m}_styles.csv')
                       .set_index('style').loc[group] for m in NAMES]
            self.assertTrue(all(s.split == 'val' for s in sources))
            self.assertEqual({int(s.n) for s in sources}, {int(row[1].replace(',', ''))})
            self.assertEqual(row[2:5], [f'{s.macro_f1:.4f}' for s in sources])
            self.assertEqual(row[5], f'{100*(sources[2].macro_f1-sources[1].macro_f1):+.2f}')

    def test_q5_thresholds_and_tradeoffs_match_saved_predictions(self):
        slide = next(s for s in self.slides if s.get('rq') == 'Q5' and 'table' in s)
        rows = slide['table']['rows']
        self.assertEqual(len(rows), 6)
        for i, model in enumerate(NAMES):
            sweep = pd.read_csv(OUT / f'analysis/27_emotion27/{model}/tables/{model}_thresholds_val.csv')
            best = sweep.loc[sweep.macro_f1.idxmax()]
            with np.load(OUT / f'evidence/27/{model}/val_predictions.npz', allow_pickle=False) as a:
                truth, probabilities = a['true_label_vector'], a['probability_vector']
                default_f1 = f1_score(truth, probabilities >= .5, average='macro', zero_division=0)
                for offset, threshold in ((0, .5), (1, float(best.threshold))):
                    row = rows[2*i+offset]
                    predicted = probabilities >= threshold
                    score = f1_score(truth, predicted, average='macro', zero_division=0)
                    loss = hamming_loss(truth, predicted)
                    self.assertEqual(row[0], NAMES[model])
                    self.assertEqual(row[2:5], [f'{threshold:.2f}', f'{score:.4f}', f'{loss:.4f}'])
                    if offset:
                        self.assertEqual(row[5], f'{100*(score-default_f1):+.3f}')

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
