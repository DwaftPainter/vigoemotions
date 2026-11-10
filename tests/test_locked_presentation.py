"""Prove styling preserves finalized content and four dedicated research tables."""
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from academic_slides import macro_values, render_deck
from build_equal20_presentation import presentation_content, script_markdown
from locked_slide_design import approved_content, design_slides, main_markdown, presentation_markdown
from import_equal20_results import OUT


class LockedPresentationTests(unittest.TestCase):
    def test_slide_14_omits_the_removed_review_commitment(self):
        self.assertNotIn('Review at least 20 false negatives and 20 false positives.', self.main[13][1])
        self.assertNotIn('twenty false negatives and twenty false positives', self.main[13][3])

    @classmethod
    def setUpClass(cls):
        cls.main = presentation_content()
        cls.design = design_slides(cls.main)

    def test_only_two_requested_slides_change_and_all_notes_are_preserved(self):
        expected = json.loads((OUT / 'slides.json').read_text())
        expected[13][1].remove('Review at least 20 false negatives and 20 false positives.')
        expected[13][3] = expected[13][3].removesuffix(' We also plan a detailed review of at least twenty false negatives and twenty false positives.')
        self.assertEqual(approved_content(self.main), expected)
        for i, (s, row) in enumerate(zip(self.design, expected), 1):
            self.assertEqual(s['notes'], row[3])
            if i in (10, 11):
                self.assertEqual(s['replaces_slide'], i)
                self.assertTrue(s['research_tables'])
            else:
                self.assertEqual([s['title'], s['bullets'], s['figure'], s['notes']], row)
                self.assertFalse(s['research_tables'])
        self.assertEqual(len(self.design), 15)
        current_md = presentation_markdown(self.main, self.design)
        self.assertNotIn('## 16.', current_md)
        baseline_sections = re.split(r'(?m)^## \d+\. ', main_markdown(expected))[1:]
        current_sections = re.split(r'(?m)^## \d+\. ', current_md)[1:]
        self.assertEqual(len(current_sections), 15)
        for i, (before, after) in enumerate(zip(baseline_sections, current_sections), 1):
            if i not in (10, 11):
                self.assertEqual(before, after)
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

    def test_q1_q2_tables_match_current_per_label_results(self):
        q1, q2 = self.design[9]['tables']
        for i, model in enumerate(('tfidf_lr', 'gru', 'phobert')):
            per = pd.read_csv(OUT / f'analysis/27_emotion27/{model}/tables/{model}_per_label.csv')
            self.assertTrue(per.split.eq('val').all())
            best, worst = per.loc[per.f1.idxmax()], per.loc[per.f1.idxmin()]
            self.assertEqual(q1['rows'][i][1:], [f'{best.label}\n{best.f1:.4f}', f'{worst.label}\n{worst.f1:.4f}'])
            rho, pvalue = spearmanr(per.train_count, per.f1)
            self.assertEqual(q2['rows'][i][1:], [f'{rho:.3f}', f'{pvalue:.3f}'])

    def test_q3_q4_tables_use_current_compatible_validation_results(self):
        pair, style = self.design[10]['tables']
        source = pd.read_csv(OUT / 'pair_preservation.csv')
        self.assertEqual(pair['rows'], [[r.model, str(r.joint_TP), str(r.val_true), f'{r.joint_recall:.2%}']
                                       for r in source.itertuples()])
        source = pd.read_csv(OUT / 'neural_style_comparison.csv')
        self.assertEqual(len(style['rows']), len(source))
        for row, run in zip(style['rows'], source.itertuples()):
            self.assertEqual(row[1:], [f'{run.n:,}', f'{run.gru:.4f}', f'{run.phobert:.4f}', f'{run.delta_pp:+.2f}'])

    def test_four_dedicated_tables_replace_supplementary_labels(self):
        questions = [table['rq'] for slide in self.design[9:11] for table in slide['tables']]
        self.assertEqual(questions, ['Q1', 'Q2', 'Q3', 'Q4'])
        self.assertNotIn('Supplementary Table', presentation_markdown(self.main, self.design))

    def test_every_main_sentence_survives_rendering_and_no_text_overlaps(self):
        normalize = lambda text: re.sub(r'\s+', '', text)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'slides.pdf'
            boxes = render_deck(self.design, target)
            pages = PdfReader(target).pages
            self.assertEqual(len(pages), 15)
            self.assertNotIn('SUPPLEMENTARY', ''.join(p.extract_text() for p in pages).upper())
            for i, (page, (title, bullets, figure, notes)) in enumerate(zip(pages, self.main)):
                text = normalize(page.extract_text())
                if i in (9, 10):
                    sentences = [self.design[i]['title']]
                    sentences += [cell for t in self.design[i]['tables'] for row in t['rows'] for cell in row]
                else:
                    sentences = [title] + bullets
                for sentence in sentences:
                    self.assertIn(normalize(sentence), text)
                self.assertEqual((float(page.mediabox.width), float(page.mediabox.height)), (1280, 720))
            for a, b in combinations(boxes, 2):
                if a['slide'] != b['slide']:
                    continue
                dx = min(a['x']+a['width'], b['x']+b['width'])-max(a['x'], b['x'])
                dy = min(a['y']+a['height'], b['y']+b['height'])-max(a['y'], b['y'])
                self.assertFalse(dx > 2 and dy > 2, f'Text overlap: {a} / {b}')
