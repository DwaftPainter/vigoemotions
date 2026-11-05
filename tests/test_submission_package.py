from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from submission_package import is_public_report, sanitized_notebook, write_public_member


class SubmissionPackageTests(unittest.TestCase):
    def test_notebook_sanitization_removes_outputs_and_widget_state_without_mutating_input(self):
        original = {'metadata': {'widgets': {'private': 'restricted text'}, 'kernelspec': {'name': 'python3'}},
            'cells': [{'cell_type': 'code', 'source': ['print(1)'], 'execution_count': 8,
                       'metadata': {'execution': {'private': 'restricted text'}},
                       'outputs': [{'text': 'restricted text'}]}]}
        before = deepcopy(original)
        clean = sanitized_notebook(original)
        self.assertEqual(original, before)
        self.assertEqual(clean['cells'][0]['outputs'], [])
        self.assertIsNone(clean['cells'][0]['execution_count'])
        self.assertNotIn('restricted text', json.dumps(clean))
        self.assertEqual(clean['metadata']['kernelspec'], {'name': 'python3'})

    def test_archived_notebook_is_clean_even_when_source_has_private_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notebook = root/'demo.ipynb'
            notebook.write_text(json.dumps({'metadata': {}, 'cells': [
                {'cell_type': 'code', 'source': ['x = 1'], 'metadata': {},
                 'execution_count': 1, 'outputs': [{'text': 'restricted text'}]}]}))
            with ZipFile(root/'code.zip', 'w') as archive:
                write_public_member(archive, notebook, root)
            with ZipFile(root/'code.zip') as archive:
                self.assertNotIn('restricted text', archive.read('demo.ipynb').decode())
            self.assertIn('restricted text', notebook.read_text())

    def test_public_report_filter_excludes_private_models_resources_and_text_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['reports/neutral_28_labels/config.json', 'reports/x/best.pt',
                         'reports/x/vocab.json', 'reports/x/tokenizer/config.json',
                         'reports/x/resources/model.json', 'reports/x/model.joblib',
                         'reports/x/prediction_examples.csv']:
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
                self.assertFalse(is_public_report(path, root), name)
            path = root/'reports/tables/private.csv'; path.parent.mkdir(parents=True)
            path.write_text('id,text,labels\n1,restricted text,[27]\n')
            self.assertFalse(is_public_report(path, root))
            path.write_text('model,macro_f1\nfixture,0.5\n')
            self.assertTrue(is_public_report(path, root))

    def test_named_configs_match_recorded_runs(self):
        baseline = json.loads((ROOT/'configs/baseline27_4ep.json').read_text())
        neutral = json.loads((ROOT/'configs/neutral28_12ep_base.json').read_text())
        self.assertEqual(baseline, json.loads((ROOT/'reports/phobert_run.json').read_text())['config'])
        self.assertEqual(neutral, json.loads((ROOT/'archive/reports/midterm_2026-11-04/evidence/study.json').read_text())['baseline_config'])
        self.assertEqual((baseline['gru']['epochs'], baseline['phobert']['epochs']), (10, 4))
        self.assertEqual((neutral['gru']['epochs'], neutral['phobert']['epochs']), (10, 12))
