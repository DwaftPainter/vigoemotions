"""Study PDFs must quote the actual result scope and preserve source artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_oral_defence import build, substitutions, verify_sources

ROOT = Path(__file__).resolve().parents[1]


class OralDefenceTests(unittest.TestCase):
    def test_key_numbers_use_all28_and_observed_test_deltas(self):
        values = substitutions()
        self.assertEqual(values['test_fixed_macro'], '0.589964')
        self.assertEqual(values['test_global_macro'], '0.591254')
        self.assertEqual(values['test_per_label_macro'], '0.585415')
        self.assertEqual(values['global_gain_pp'], '+0.1289')
        self.assertEqual(values['per_label_gain_pp'], '-0.4549')
        self.assertNotIn('0.595755', values['primary_table'])
        self.assertIn('0.595755', values['emotion27_table'])

    def test_rq4_results_compare_noise_against_the_same_fixed_control(self):
        values = substitutions()
        self.assertIn('0.589964', values['rq4_table'])
        self.assertIn('0.329749', values['rq4_table'])
        self.assertIn('0.550578', values['rq4_table'])
        self.assertEqual(values['rq4_diacritics_drop_pp'], '26.02')
        self.assertEqual(values['rq4_typo_drop_pp'], '3.94')
        self.assertEqual(values['rq4_diacritics_tail_drop_pp'], '31.53')
        self.assertEqual(values['rq4_typo_tail_drop_pp'], '5.41')

    def test_changed_retained_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'docs/oral_defence').mkdir(parents=True)
            sample = root/'source.csv'; sample.write_text('verified\n')
            manifest = {'retained_original_sha256': {'source.csv': hashlib.sha256(sample.read_bytes()).hexdigest()}}
            (root/'docs/oral_defence/source_manifest.json').write_text(json.dumps(manifest))
            verify_sources(root)
            sample.write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'source'):
                verify_sources(root)

    def test_five_complete_notes_export_without_touching_evidence(self):
        before = verify_sources()
        with tempfile.TemporaryDirectory() as tmp:
            result = build(Path(tmp))
            self.assertEqual(len(result['documents']), 5)
            for item in result['documents']:
                md = Path(tmp)/item['markdown']
                pdf = Path(tmp)/item['pdf']
                self.assertNotIn('{{', md.read_text())
                text = '\n'.join(p.extract_text() or '' for p in PdfReader(pdf).pages)
                self.assertNotIn('\ufffd', text)
                self.assertIn('ViGoEmotions', text)
                self.assertGreater(item['pages'], 0)
                self.assertEqual(hashlib.sha256(pdf.read_bytes()).hexdigest(), item['pdf_sha256'])
            questions = (Path(tmp)/'03_questions_answers_vi.md').read_text()
            self.assertEqual(questions.count('**English:**'), 34)
            self.assertIn('### 34.', questions)
            for prefix in ('00', '02', '03', '04'):
                note = next(item for item in result['documents'] if item['markdown'].startswith(prefix))
                pdf_text = '\n'.join(p.extract_text() or '' for p in PdfReader(Path(tmp)/note['pdf']).pages)
                self.assertIn('RQ4', pdf_text)
            self.assertIn('reports/rq4_noise/test/tables/comparison.csv', result['rq4_source_sha256'])
            practice = (Path(tmp)/'04_practice_vi.md').read_text()
            self.assertIn('5/9', practice)
            self.assertFalse(result['model_inference_performed'])
        self.assertEqual(before, verify_sources())


if __name__ == '__main__':
    unittest.main()
