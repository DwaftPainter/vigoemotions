"""Exercise notebook 06 Kaggle I/O offline with temporary Input/Working folders."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_kaggle as kaggle


class KaggleNeutral28Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        self.input = self.folder / 'input'
        self.input.mkdir()
        self.root = self.folder / 'working/ViGoEmotions28'
        self.root.parent.mkdir()
        target = kaggle.build_neutral28(self.folder / '06.ipynb')
        self.notebook = json.loads(target.read_text())
        self.namespace = {}

    def execute(self, index):
        source = ''.join(self.notebook['cells'][index]['source'])
        source = source.replace('/kaggle/input', str(self.input))
        source = source.replace('/kaggle/working', str(self.root.parent))
        exec(compile(source, f'cell_{index}', 'exec'), self.namespace)

    def archive(self, member='scripts/neutral_28_labels.py'):
        with ZipFile(self.input / 'vigoemotions_neutral28_colab.zip', 'w') as archive:
            archive.writestr(member, '# fixture module\n')

    def csvs(self):
        for name in ('train.csv', 'val.csv', 'test.csv'):
            (self.input / name).write_text('id,text,labels\n1,fixture,[27]\n')

    def test_generated_notebook_has_valid_python_and_no_colab_runtime(self):
        for cell in self.notebook['cells']:
            source = ''.join(cell['source'])
            if cell['cell_type'] == 'code':
                ast.parse(source)
                self.assertNotIn('google.colab', source)
                self.assertNotIn('/content/', source)
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_extracts_zip_and_refuses_to_overwrite_existing_work(self):
        self.archive()
        self.execute(2)
        self.assertTrue((self.root / 'scripts/neutral_28_labels.py').is_file())
        with self.assertRaises(FileExistsError):
            self.execute(2)

    def test_accepts_dataset_that_kaggle_has_already_extracted(self):
        module = self.input / 'code/scripts/neutral_28_labels.py'
        module.parent.mkdir(parents=True)
        module.write_text('# fixture module\n')
        self.execute(2)
        self.assertEqual((self.root / 'scripts/neutral_28_labels.py').read_text(), module.read_text())

    def test_rejects_zip_path_escape_before_extracting(self):
        self.archive('../escaped.txt')
        with self.assertRaisesRegex(ValueError, 'Unsafe ZIP path'):
            self.execute(2)
        self.assertFalse((self.root.parent / 'escaped.txt').exists())

    def test_csvs_are_copied_to_working_and_data_variable_is_available(self):
        self.archive()
        self.csvs()
        self.execute(2)
        self.execute(8)
        self.assertEqual(self.namespace['DATA'], self.root / 'data')
        for name in ('train.csv', 'val.csv', 'test.csv'):
            self.assertEqual((self.root / 'data' / name).read_bytes(), (self.input / name).read_bytes())

    def test_missing_duplicate_and_invalid_csv_fail_before_copying(self):
        self.archive()
        self.execute(2)
        with self.assertRaises(RuntimeError):
            self.execute(8)
        self.csvs()
        duplicate = self.input / 'extra/train.csv'
        duplicate.parent.mkdir()
        duplicate.write_text('id,text,labels\n')
        with self.assertRaises(RuntimeError):
            self.execute(8)
        duplicate.unlink()
        (self.input / 'test.csv').write_text('wrong,header\n')
        with self.assertRaises(ValueError):
            self.execute(8)
        self.assertFalse((self.root / 'data/train.csv').exists())

    def test_export_includes_study_checkpoint_but_excludes_source_data(self):
        out = self.root / 'reports/neutral_28_labels'
        (out / 'models/gru').mkdir(parents=True)
        (out / 'models/gru/best.pt').write_bytes(b'checkpoint')
        (out / 'metrics_summary.csv').write_text('model,macro_f1\n')
        (self.root / 'data').mkdir()
        (self.root / 'data/train.csv').write_text('private raw text')
        self.namespace.update(OUT=out, Path=Path)
        self.execute(28)
        with ZipFile(self.root.parent / 'vigoemotions_neutral28_private_results.zip') as archive:
            self.assertEqual(set(archive.namelist()), {
                'neutral_28_labels/models/gru/best.pt',
                'neutral_28_labels/metrics_summary.csv',
            })


if __name__ == '__main__':
    unittest.main()
