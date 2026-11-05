from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import kaggle_equal_epochs as paired
import neutral_28_labels as neutral


class EqualEpochTests(unittest.TestCase):
    def test_canonical_profile_has_equal_neural_budget_and_shared_selection(self):
        config = paired.load_profile(ROOT/'configs/kaggle_equal_20ep.json')
        self.assertEqual((config['gru']['epochs'], config['phobert']['epochs']), (20, 20))
        cfg28 = neutral.study_config(config)
        self.assertEqual(cfg28['label_count'], 28)
        for key in ['gru', 'phobert', 'seed', 'threshold', 'tfidf', 'logistic_regression']:
            self.assertEqual(config[key], cfg28[key])
        self.assertEqual(cfg28['checkpoint_selection_metric'], 'emotion27_macro_f1')
        self.assertIn('27', cfg28['checkpoint_selection'])

    def test_rejects_unequal_epochs_or_different_selection(self):
        original = json.loads((ROOT/'configs/kaggle_equal_20ep.json').read_text())
        for key, value in [('gru', 10), ('phobert', 12)]:
            config = deepcopy(original); config[key]['epochs'] = value
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp)/'profile.json'; p.write_text(json.dumps(config))
                with self.assertRaises(ValueError): paired.load_profile(p)
        original['checkpoint_selection_metric'] = 'macro_f1'
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'profile.json'; p.write_text(json.dumps(original))
            with self.assertRaises(ValueError): paired.load_profile(p)

    def test_checkpoint_criterion_ignores_neutral_column_only_in_paired_profile(self):
        y = np.zeros((2, 28), dtype=int); y[0, 0] = 1; y[1, 27] = 1
        p = y * .8 + .1
        config = paired.load_profile(ROOT/'configs/kaggle_equal_20ep.json')
        first = neutral.checkpoint_score(config, y, p)
        p[:, 27] = 1 - p[:, 27]
        self.assertEqual(first, neutral.checkpoint_score(config, y, p))
        self.assertNotEqual(neutral.checkpoint_score({}, y, y*.8+.1), neutral.checkpoint_score({}, y, p))

    def notebook_paths(self):
        return {27: ROOT/'notebooks/08_kaggle_equal_20ep_27_labels.ipynb',
                28: ROOT/'notebooks/09_kaggle_equal_20ep_28_labels.ipynb'}

    def test_separate_notebooks_run_only_their_three_models(self):
        for labels,path in self.notebook_paths().items():
            notebook=json.loads(path.read_text())
            code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
            self.assertIn(f'LABEL_COUNT = {labels}',code)
            self.assertIn("'--labels', str(LABEL_COUNT)",code)
            self.assertEqual(code.count("run('train'"),3)
            for model in paired.MODELS:self.assertIn(f"run('train', '--model', '{model}')",code)
            self.assertIn("status['expected_runs'] == 3",code)
            self.assertIn(f'vigoemotions_kaggle_equal20_{labels}labels_private_results.zip',code)
            self.assertNotIn("run('test'",code)
            self.assertTrue(all(not c.get('outputs') for c in notebook['cells']))

    def test_kaggle_bootstrap_and_csv_discovery_work_offline(self):
        from zipfile import ZipFile
        import os
        previous=Path.cwd()
        for labels,path in self.notebook_paths().items():
            notebook=json.loads(path.read_text())
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    base=Path(tmp);inputs=base/'input';inputs.mkdir();working=base/'working';working.mkdir()
                    with ZipFile(inputs/'vigoemotions_kaggle_equal_20ep.zip','w') as z:
                        z.writestr('scripts/kaggle_equal_epochs.py','# synthetic code fixture')
                    source=''.join(notebook['cells'][2]['source']).replace('/kaggle/input',str(inputs)).replace('/kaggle/working',str(working))
                    namespace={};exec(compile(source,'bootstrap','exec'),namespace)
                    expected=working/f'ViGoEmotions{labels}Equal20'
                    self.assertTrue((expected/'scripts/kaggle_equal_epochs.py').exists())
                    for name in ['train.csv','val.csv','test.csv']:
                        (inputs/name).write_text('id,text,labels\nfixture,synthetic,[27]\n')
                        ids=inputs/'code/splits'/name;ids.parent.mkdir(parents=True,exist_ok=True);ids.write_text('id\nfixture\n')
                    data=''.join(next(c['source'] for c in notebook['cells'] if 'DATA_DIR = ROOT/' in ''.join(c['source'])))
                    data=data.replace('/kaggle/input',str(inputs));exec(compile(data,'data','exec'),namespace)
                    self.assertEqual(namespace['DATA_DIR'],expected/'data')
                    for name in ['train.csv','val.csv','test.csv']:
                        self.assertEqual((namespace['DATA_DIR']/name).read_bytes(),(inputs/name).read_bytes())
                    with self.assertRaises(FileExistsError):exec(compile(source,'bootstrap','exec'),namespace)
                    os.chdir(previous)
            finally:os.chdir(previous)

    def test_notebook_python_cells_have_valid_syntax(self):
        import ast
        for path in self.notebook_paths().values():
            notebook=json.loads(path.read_text())
            for i,cell in enumerate(notebook['cells']):
                if cell['cell_type']=='code':
                    source=''.join(cell['source'])
                    ast.parse('\n'.join(line for line in source.splitlines() if not line.startswith(('%','!'))),filename=f'cell_{i}')

    def test_missing_run_results_are_not_filled_from_legacy_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            table = paired.result_rows(path, {'val': pd.DataFrame()})
            self.assertTrue(table.empty)

    def test_empty_or_invalid_label_scope_is_not_a_completed_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for spaces in ([],[29],[27,29]):
                (out/'manifest.json').write_text(json.dumps({'label_spaces':spaces}))
                with self.assertRaisesRegex(ValueError,'scope'):paired.result_rows(out,{'val':pd.DataFrame()})

    def recorded_fixture(self, out, labels, model):
        config = paired.load_profile(ROOT/'configs/kaggle_equal_20ep.json')
        (out/'config27.json').write_text(json.dumps(config))
        frame = pd.DataFrame([dict(id='fixture-a',label_ids=[2]),dict(id='fixture-b',label_ids=[27])])
        y = neutral.targets28(frame)
        if labels == 27: y = y[:, :27]
        p = y * .8 + .1
        common_score = neutral.scores(y[:, :27], p[:, :27])['macro_f1']
        folder, metadata, prediction = paired._locations(out, labels, model)
        prediction.parent.mkdir(parents=True, exist_ok=True)
        np.savez(prediction, sample_id=frame.id.to_numpy(dtype=str), true_label_vector=y,
                 probability_vector=p, labels=np.array(neutral.LABELS28[:labels]))
        if labels == 28: config = neutral.study_config(config)
        run = dict(config=config, parameter_count=1,train_seconds=1,device='synthetic fixture',best_epoch=3,
            history=[dict(epoch=i,macro_f1=common_score if i==3 else 0,
                          selection_macro_f1=common_score if i==3 else 0) for i in range(1,21)])
        metadata.write_text(json.dumps(run))
        checkpoint = folder/'model.joblib' if model=='tfidf_lr' else (
            folder/'pretrained/model.safetensors' if labels==28 and model=='phobert' else folder/'best.pt')
        checkpoint.parent.mkdir(parents=True,exist_ok=True); checkpoint.write_bytes(b'synthetic fixture')
        return {'val':frame}, metadata

    def test_summary_verifies_all_six_fixture_runs_and_keeps_tfidf_epoch_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for labels in [27,28]:
                for model in paired.MODELS: frames,_=self.recorded_fixture(out,labels,model)
            table=paired.result_rows(out,frames)
            self.assertEqual(len(table),6)
            self.assertTrue(table.query("model=='tfidf_lr'").epochs_trained.isna().all())
            self.assertEqual(table.query("model!='tfidf_lr'").epochs_trained.tolist(),[20]*4)

    def test_each_label_scope_completes_independently_after_three_models(self):
        from unittest.mock import patch
        for labels in [27,28]:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                for model in paired.MODELS:frames,_=self.recorded_fixture(out,labels,model)
                (out/'manifest.json').write_text(json.dumps({'label_spaces':[labels]}))
                with patch.object(paired,'checked',return_value=(out,{},frames)):
                    rows=paired.report(out,ROOT/'data')
                    with self.assertRaisesRegex(ValueError,'scoped'):
                        paired.train(out,ROOT/'data',55-labels,'tfidf_lr')
                status=json.loads((out/'status.json').read_text())
                self.assertEqual(status['status'],'completed')
                self.assertEqual(status['expected_runs'],3)
                self.assertEqual(status['completed_runs'],3)
                self.assertEqual(rows.label_count.tolist(),[labels]*3)

    def test_summary_refuses_short_history_or_foreign_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);frames,path=self.recorded_fixture(out,28,'gru')
            original=json.loads(path.read_text())
            bad=deepcopy(original);bad['history']=bad['history'][:19];path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'Incomplete'):paired.result_rows(out,frames)
            bad=deepcopy(original);bad['config']['seed']=999;path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'frozen'):paired.result_rows(out,frames)

    def test_summary_refuses_different_validation_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);frames,_=self.recorded_fixture(out,27,'gru')
            frames['val']=frames['val'].iloc[::-1]
            with self.assertRaises(AssertionError):paired.result_rows(out,frames)

    def test_read_only_report_keeps_data_checks_but_training_locks_runtime(self):
        from unittest.mock import patch
        import hashlib
        config=paired.load_profile(ROOT/'configs/kaggle_equal_20ep.json')
        frame=pd.DataFrame({'id':['synthetic']})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'config27.json').write_text(json.dumps(config))
            (out/'manifest.json').write_text(json.dumps(dict(profile=config,
                environment={'data_sha256':{'val':'fixture'},'gpu':'original'},source_sha256={},
                split_ids_sha256={'val':hashlib.sha256(b'synthetic').hexdigest()})))
            with patch.object(paired,'study_path',return_value=out), \
                 patch.object(neutral,'runtime',return_value={'data_sha256':{'val':'fixture'}}), \
                 patch.object(paired,'runtime_identity',return_value={'data_sha256':{'val':'fixture'},'gpu':'different'}), \
                 patch.object(neutral,'load_frames',return_value=({'val':frame},None,None)):
                self.assertEqual(paired.checked(out,ROOT/'data')[0],out)
                with self.assertRaisesRegex(ValueError,'Runtime'):paired.checked(out,ROOT/'data',lock_runtime=True)

    def test_tfidf_custom_output_preserves_historical_reports(self):
        from unittest.mock import patch
        import train_tfidf
        frame = pd.DataFrame([dict(id=str(i),text=f'fixture word{i} common',labels=str([i]),
            label_ids=[i],normalized=f'fixture word{i} common') for i in range(28)])
        frames = {key: frame for key in ('train','val','test')}
        config = paired.load_profile(ROOT/'configs/kaggle_equal_20ep.json')
        before = (ROOT/'reports/tfidf_lr_run.json').read_bytes() if (ROOT/'reports/tfidf_lr_run.json').exists() else None
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(train_tfidf, 'load_data', return_value=(frames, None)), \
             patch.object(train_tfidf, 'environment', return_value={'synthetic_fixture': True}):
            path = Path(tmp)/'isolated'
            train_tfidf.main(config=config, output_dir=path)
            self.assertTrue((path/'model.joblib').exists())
            self.assertEqual(json.loads((path/'tfidf_lr_run.json').read_text())['config'], config)
            with self.assertRaises(FileExistsError): train_tfidf.main(config=config, output_dir=path)
        if before is not None: self.assertEqual((ROOT/'reports/tfidf_lr_run.json').read_bytes(), before)
