"""Prove styling preserves the finalized presentation and optional table scopes."""
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import json
import re
import sys
import tempfile
import unittest

import pandas as pd
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from academic_slides import macro_values, render_deck
from build_equal20_presentation import presentation_content, script_markdown
from locked_slide_design import approved_content, design_slides, main_markdown, presentation_markdown
from import_equal20_results import OUT


class LockedPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = presentation_content()
        cls.design = design_slides(cls.main)

    def test_all_main_content_and_speaker_notes_match_approved_snapshot(self):
        expected = json.loads((OUT / 'slides.json').read_text())
        self.assertEqual(approved_content(self.main), expected)
        for s, row in zip(self.design[:15], expected):
            self.assertEqual([s['title'], s['bullets'], s['figure'], s['notes']], row)
            self.assertFalse(s['supplemental'])
        self.assertEqual(len(self.design), 17)
        self.assertTrue(all(s['supplemental'] for s in self.design[15:]))
        self.assertTrue(presentation_markdown(self.main, self.design).startswith(main_markdown(expected)))
        self.assertEqual(script_markdown(self.main), script_markdown(expected))

    def test_content_lock_rejects_edits_even_when_numbers_do_not_change(self):
        for field, replacement in ((0, 'New heading'), (1, ['Shorter wording']),
                                    (2, 'new_figure.png'), (3, 'Rewritten notes')):
            altered = deepcopy(json.loads(json.dumps(self.main)))
            altered[0][field] = replacement
            with self.assertRaisesRegex(ValueError, 'locked'):
                design_slides(altered)

    def test_main_charts_only_restate_the_approved_metric_values(self):
        comparison = pd.read_csv(OUT / 'model_comparison.csv')
        for index, labels in ((4, 27), (5, 28)):
            plotted = macro_values(self.main[index][1])
            expected = comparison[comparison.label_count == labels].macro_f1.tolist()
            self.assertEqual(plotted, [float(f'{v:.4f}') for v in expected])

    def test_appendix_tables_use_current_compatible_validation_results(self):
        comparison = pd.read_csv(OUT / 'model_comparison.csv')
        rows = self.design[15]['tables'][0]['rows']
        self.assertEqual(len(rows), 6)
        for row, run in zip(rows, comparison.itertuples()):
            self.assertEqual(int(row[0]), run.label_count)
            self.assertEqual(run.split, 'val')
            self.assertEqual(row[2:6], [f'{getattr(run, key):.4f}' for key in
                                      ('macro_f1', 'micro_f1', 'hamming_loss', 'lrap')])
        pair, style = self.design[16]['tables']
        source = pd.read_csv(OUT / 'pair_preservation.csv')
        self.assertEqual(pair['rows'], [[r.model, str(r.joint_TP), str(r.val_true), f'{r.joint_recall:.2%}']
                                       for r in source.itertuples()])
        source = pd.read_csv(OUT / 'neural_style_comparison.csv')
        self.assertEqual(len(style['rows']), len(source))
        for row, run in zip(style['rows'], source.itertuples()):
            self.assertEqual(row[1:], [f'{run.n:,}', f'{run.gru:.4f}', f'{run.phobert:.4f}', f'{run.delta_pp:+.2f}'])

    def test_every_main_sentence_survives_rendering_and_no_text_overlaps(self):
        normalize = lambda text: re.sub(r'\s+', '', text)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'slides.pdf'
            boxes = render_deck(self.design, target)
            pages = PdfReader(target).pages
            self.assertEqual(len(pages), 17)
            for page, (title, bullets, figure, notes) in zip(pages, self.main):
                text = normalize(page.extract_text())
                for sentence in [title] + bullets:
                    self.assertIn(normalize(sentence), text)
                self.assertEqual((float(page.mediabox.width), float(page.mediabox.height)), (1280, 720))
            for a, b in combinations(boxes, 2):
                if a['slide'] != b['slide']:
                    continue
                dx = min(a['x']+a['width'], b['x']+b['width'])-max(a['x'], b['x'])
                dy = min(a['y']+a['height'], b['y']+b['height'])-max(a['y'], b['y'])
                self.assertFalse(dx > 2 and dy > 2, f'Text overlap: {a} / {b}')
