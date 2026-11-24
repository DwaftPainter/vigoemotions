from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from rq4_noise import remove_diacritics, qwerty_typos, prepare_conditions


class TextNoiseTests(unittest.TestCase):
    def test_diacritic_removal_handles_decomposed_text_d_stroke_and_case(self):
        self.assertEqual(remove_diacritics('Tiếng Việt, ĐẸP quá! 😊'), 'Tieng Viet, DEP qua! 😊')
        self.assertEqual(remove_diacritics('naïve — café ✨'), 'naïve — cafe ✨')

    def test_typos_are_reproducible_and_preserve_case_and_non_ascii(self):
        text = 'A u ă, Đ! 😊'
        result = qwerty_typos(text, sample_id='heldout-1', seed=42, rate=1)
        self.assertEqual(result, qwerty_typos(text, sample_id='heldout-1', seed=42, rate=1))
        self.assertTrue(result[0].isupper())
        self.assertIn(result[2], 'yijh')
        self.assertEqual(result[3:], ' ă, Đ! 😊')
        self.assertEqual(qwerty_typos(text, sample_id='heldout-1', seed=42, rate=0), text)

    def test_conditions_preserve_sample_order_and_labels_and_report_noops(self):
        frame = pd.DataFrame({'id':['b','a'], 'normalized':['đẹp!', '🙂'], 'labels':['0;2','27']})
        conditions, counts = prepare_conditions(frame, seed=42, rate=1)
        self.assertEqual(list(conditions), ['original','no_diacritics','qwerty_typo'])
        self.assertEqual(conditions['no_diacritics'].normalized.tolist(), ['dep!','🙂'])
        for variant in conditions.values():
            self.assertEqual(variant.id.tolist(), ['b','a'])
            self.assertEqual(variant.labels.tolist(), ['0;2','27'])
        self.assertEqual(counts['no_diacritics']['changed_samples'], 1)
        self.assertEqual(counts['no_diacritics']['edited_characters'], 2)
        self.assertEqual(counts['qwerty_typo']['eligible_characters'], 1)
        self.assertEqual(frame.normalized.tolist(), ['đẹp!','🙂'])


if __name__ == '__main__':
    unittest.main()
