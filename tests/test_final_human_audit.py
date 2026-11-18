"""Deterministic audit sampling and honest human-review completion checks."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from final_human_audit import select_pairs, build_cases, review_packet, validate_reviews, preserve_packet, private_directory
from neutral_28_labels import LABELS28


class FinalHumanAuditTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
        self.y = np.zeros((6, 28), dtype=int)
        self.y[0, [0, 1, 2, 27]] = 1
        self.y[1, [0, 1, 2]] = 1
        self.y[2, [1, 2]] = 1
        self.y[3, 27] = 1
        self.y[4, [0, 2]] = 1
        self.y[5, [0, 1]] = 1
        pred = np.zeros((6, 28), dtype=bool)
        pred[0, [0, 1, 2]] = True
        pred[1, [0, 1, 2, 27]] = True
        pred[2, [0, 1]] = True
        pred[4, [0, 1, 2]] = True
        self.values = dict(sample_id=self.ids, labels=np.array(LABELS28), true_label_vector=self.y,
                           probability_vector=pred*.8+.1, pred_fixed=pred.copy(),
                           pred_global=pred.copy(), pred_per_label=pred.copy(),
                           label_cardinality=self.y.sum(axis=1), emotion_cardinality=self.y[:, :27].sum(axis=1))
        self.fitted = dict(global_threshold=.4, per_label_thresholds=[.5]*28)
        # All train co-occurrence counts tie; canonical pair IDs must break ties.
        self.train = np.zeros((2, 28), dtype=int)
        self.train[:, :3] = 1
        self.cases, self.design = build_cases(self.values, self.fitted, self.train)
        self.texts = {sample: f'Private synthetic comment {sample}' for sample in self.ids}
        self.packet = review_packet(self.cases, self.texts)

    def test_sampling_prioritizes_comment_diversity_and_is_order_independent(self):
        y = np.array([[1, 1, 1], [1, 1, 1]])
        pred = np.zeros_like(y, dtype=bool)
        chosen = select_pairs(np.array(['b', 'a']), y, pred, 'FN', 4)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(len({i for i, c in chosen[:2]}), 2)
        a = [('b' if i == 0 else 'a', c) for i, c in chosen]
        reverse = select_pairs(np.array(['a', 'b']), y, pred, 'FN', 4)
        b = [('a' if i == 0 else 'b', c) for i, c in reverse]
        self.assertEqual(a, b)

    def test_insufficient_error_population_is_exhausted_without_duplicates(self):
        chosen = select_pairs(self.ids, self.y, self.values['pred_per_label'], 'FN', 20)
        self.assertEqual(len(chosen), len(np.argwhere((self.y == 1) & ~self.values['pred_per_label'])))
        self.assertEqual(len(chosen), len(set(chosen)))

    def test_core_cases_match_per_label_errors_and_public_rows_have_no_text(self):
        for kind in ('FN', 'FP'):
            rows = self.cases[self.cases.selection_group == f'core_{kind}']
            self.assertGreater(len(rows), 0)
            self.assertTrue(rows.per_label_outcome.eq(kind).all())
        self.assertNotIn('text', self.cases.columns)
        self.assertFalse(any('Private synthetic comment' in str(v) for v in self.cases.to_numpy().flat))

    def test_pairs_are_selected_from_train_and_case_comments_deduplicated(self):
        self.assertEqual([(p['label_id_a'], p['label_id_b']) for p in self.design['cooccurrence_pairs']], [(0, 1), (0, 2), (1, 2)])
        rows = self.cases[self.cases.selection_group.str.startswith('pair_')]
        self.assertFalse(rows.sample_id.duplicated().any())
        self.assertNotIn('neutral', ';'.join(p['label_a']+';'+p['label_b'] for p in self.design['cooccurrence_pairs']))

    def test_neutral_mixed_and_high_cardinality_are_covered(self):
        self.assertTrue(self.cases.target_label.eq('neutral').any())
        self.assertTrue(self.cases.mixed_neutral.eq('yes').any())
        rows = self.cases[self.cases.selection_group == 'high_cardinality']
        self.assertEqual(rows.sample_id.tolist(), ['a'])
        self.assertTrue(rows.label_cardinality.astype(int).ge(4).all())

    def test_blank_packet_is_pending_and_never_counts_as_human_review(self):
        result = validate_reviews(self.cases, self.packet, self.texts)
        self.assertEqual(result['completed_core_fn'], 0)
        self.assertEqual(result['completed_core_fp'], 0)
        self.assertFalse(result['audit_complete'])

    def test_completed_review_requires_identity_observation_and_private_text_confirmation(self):
        for key, value in [('reviewer', ''), ('observation', ''), ('read_private_text', ''), ('reviewer', 'ChatGPT'), ('reviewed_on', '2026-99-12')]:
            packet = self.complete_packet()
            packet.loc[0, key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_reviews(self.cases, packet, self.texts)

    def test_changed_targets_or_private_text_cannot_enter_completed_audit(self):
        for key in ('true_labels', 'text'):
            packet = self.complete_packet()
            packet.loc[0, key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_reviews(self.cases, packet, self.texts)

    def test_duplicate_review_ids_are_rejected(self):
        packet = self.packet.copy()
        packet.loc[1, 'review_id'] = packet.loc[0, 'review_id']
        with self.assertRaises(ValueError):
            validate_reviews(self.cases, packet, self.texts)

    def test_completion_counts_core_pairs_and_all_required_context_cases(self):
        result = validate_reviews(self.cases, self.complete_packet(), self.texts)
        self.assertTrue(result['audit_complete'])
        self.assertGreater(result['fn_population_shortfall'], 0)
        self.assertGreater(result['fp_population_shortfall'], 0)
        self.assertEqual(result['completed_rows'], len(self.cases))
        packet = self.complete_packet()
        context = packet.index[packet.selection_group == 'high_cardinality'][0]
        packet.loc[context, 'status'] = 'pending'
        self.assertFalse(validate_reviews(self.cases, packet, self.texts)['audit_complete'])

    def test_repreparing_packet_preserves_real_review_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'review.csv'
            self.complete_packet().to_csv(path, index=False, encoding='utf-8-sig')
            before = path.read_bytes()
            preserve_packet(path, self.packet)
            self.assertEqual(path.read_bytes(), before)

    def test_private_directory_rejects_a_public_repository_path(self):
        with self.assertRaises(ValueError):
            private_directory(Path(__file__).resolve().parents[1] / 'reports/final28/human_audit')

    def test_no_errors_or_context_cases_reports_zero_population_honestly(self):
        values = {key: value[:1].copy() for key, value in self.values.items()}
        values['true_label_vector'][:] = 0
        values['true_label_vector'][0, 0] = 1
        for strategy in ('fixed', 'global', 'per_label'):
            values['pred_' + strategy] = values['true_label_vector'].astype(bool)
        cases, design = build_cases(values, self.fitted, self.train)
        self.assertEqual(len(cases), 0)
        packet = review_packet(cases, self.texts)
        result = validate_reviews(cases, packet, self.texts)
        self.assertEqual(result['completed_rows'], 0)
        self.assertEqual(result['fn_population_shortfall'], 20)
        self.assertEqual(result['fp_population_shortfall'], 20)

    def test_private_spreadsheet_text_is_literal_and_never_a_formula(self):
        texts = {sample: '=1+1' for sample in self.ids}
        packet = review_packet(self.cases, texts)
        self.assertTrue(packet.text.eq("'=1+1").all())
        self.assertEqual(validate_reviews(self.cases, packet, texts)['completed_rows'], 0)

    def complete_packet(self):
        packet = self.packet.copy()
        for key, value in dict(reviewer='synthetic_human_fixture', status='complete', read_private_text='yes',
                               reviewed_on='2026-11-16', error_category='unclear', reference_assessment='consistent',
                               observation='Synthetic test observation; not a real audit.', hypothesis='Not evaluated in a synthetic fixture.').items():
            packet[key] = value
        return packet


if __name__ == '__main__':
    unittest.main()
