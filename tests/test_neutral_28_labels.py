"""Offline smoke tests: real tiny training/reloading; no pretrained downloads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import neutral_28_labels as study
from common import multihot, write_json


def fixture_frames():
    # Train has both classes for every label; test targets are deliberately unused.
    rows = [dict(id=f'x{i}', text=f'word{i} vui buồn', labels=str([i]),
                 label_ids=[i], normalized=f'word{i} vui buồn') for i in range(28)]
    rows.append(dict(id='mixed', text='vui bình thường', labels='[2,27,2]',
                     label_ids=[2, 27, 2], normalized='vui bình thường'))
    return pd.DataFrame(rows)


class Neutral28Tests(unittest.TestCase):
    def test_targets_preserve_original_and_27_pipeline_unchanged(self):
        y = study.multihot28([27])
        self.assertEqual(y.shape, (28,))
        self.assertEqual(y[27], 1)
        self.assertEqual(y[:27].sum(), 0)
        y = study.multihot28('[2,27,2]')
        self.assertEqual(y.sum(), 2)
        self.assertEqual(y[2], 1)
        self.assertEqual(y[27], 1)
        self.assertEqual(multihot([27]).sum(), 0)
        np.testing.assert_array_equal(multihot([2,27]), y[:27])

    def test_reject_invalid_annotations(self):
        for labels in ([], [28], [-1], [True], ['27']):
            with self.assertRaises(ValueError):
                study.multihot28(labels)

    def test_prediction_categories_and_top5_come_from_probabilities(self):
        p = np.full((4, 28), .1)
        p[0, 27] = .8
        p[1, 2] = .7
        p[2, [2, 3]] = .9
        p[3, [2, 27]] = .6
        summary, detail = study.prediction_tables(['a', 'b', 'c', 'd'], p)
        self.assertEqual(summary.neutral_predicted.tolist(), [True, False, False, True])
        self.assertEqual(summary.iloc[0].predicted_emotions, '')
        self.assertEqual(summary.iloc[2].predicted_emotions, 'joy;love')
        self.assertEqual(detail.groupby('input_index').top5.sum().tolist(), [5]*4)
        empty, detail = study.prediction_tables(['none'], np.full((1, 28), .1))
        self.assertEqual(empty.iloc[0].predicted_probabilities, '{}')
        self.assertEqual(len(detail), 5)

    def test_output_cannot_target_original_reports(self):
        for path in (study.ROOT/'reports', study.ROOT/'data/checkpoints', study.ROOT):
            with self.assertRaises(ValueError):
                study.output_path(path)

    def test_loader_preserves_annotations_and_removes_only_train_overlap(self):
        with tempfile.TemporaryDirectory(dir=study.ROOT) as tmp:
            folder = Path(tmp)
            for split, rows in {
                'train': [('a', 'trùng', '[27]'), ('b', 'vui', '[2,27]')],
                'val': [('c', 'khác', '[3]')], 'test': [('d', 'trùng', '[27]')]
            }.items():
                pd.DataFrame(rows, columns=['id','text','labels']).to_csv(folder/f'{split}.csv', index=False)
            before = {p.name: study.digest(p) for p in folder.glob('*.csv')}
            frames, excluded, counts = study.load_frames(folder)
            self.assertEqual(excluded.id.tolist(), ['a'])
            self.assertEqual(frames['train'].label_ids.tolist(), [[2, 27]])
            self.assertEqual(counts.neutral_only.sum(), 2)
            self.assertEqual(counts.mixed_neutral.sum(), 1)
            self.assertEqual(before, {p.name: study.digest(p) for p in folder.glob('*.csv')})

    def test_metrics_include_neutral_and_emotion_projection(self):
        with tempfile.TemporaryDirectory(dir=study.ROOT) as tmp:
            frame = fixture_frames()
            y = study.targets28(frame)
            p = y * .8 + .1
            row = study.save_evaluation(Path(tmp), 'fixture', 'val', frame, p)
            self.assertEqual(row['macro_f1'], 1)
            self.assertEqual(row['emotion27_macro_f1'], 1)
            table = pd.read_csv(Path(tmp)/'models/fixture/val_per_class.csv')
            self.assertEqual(len(table), 28)
            self.assertEqual(table.iloc[27]['label'], 'neutral')
            self.assertEqual(table.iloc[27].support, 2)
            self.assertEqual(table.iloc[27].predicted_positive, 2)

    def test_tfidf_fit_and_saved_reload_match(self):
        frame = fixture_frames()
        with tempfile.TemporaryDirectory(dir=study.ROOT) as tmp, patch.object(study, 'DEFAULT_OUTPUT', Path(tmp)):
            out = Path(tmp)
            folder = out/'models/tfidf_lr'
            folder.mkdir(parents=True)
            cfg = study.study_config(study.read_json(study.ROOT/'configs/midterm.json'))
            p, meta = study.train_tfidf(frame, frame, cfg, folder)
            write_json(folder/'run.json', dict(config=cfg, labels=study.LABELS28))
            loaded = study.load_predictor(out, 'tfidf_lr')
            np.testing.assert_allclose(p, loaded(frame.text.tolist()))
            self.assertEqual(p.shape, (len(frame), 28))

    def test_gru_real_backward_checkpoint_and_reload(self):
        import torch
        torch.set_num_threads(1)
        frame = fixture_frames().iloc[[0, 2, 27, 28]].copy()
        with tempfile.TemporaryDirectory(dir=study.ROOT) as tmp, patch.object(study, 'DEFAULT_OUTPUT', Path(tmp)):
            out = Path(tmp)
            folder = out/'models/gru'
            folder.mkdir(parents=True)
            cfg = study.study_config(study.read_json(study.ROOT/'configs/midterm.json'))
            cfg['gru'].update(epochs=1, embedding_dim=8, hidden_dim=8, batch_size=2, max_length=8)
            p, meta = study.train_neural28('gru', frame, frame, cfg, folder, out/'unused', allow_cpu=True)
            self.assertEqual(meta['trainable_parameter_count'], meta['parameter_count'])
            self.assertEqual(meta['best_epoch'], 1)
            write_json(folder/'run.json', dict(config=cfg, labels=study.LABELS28))
            loaded = study.load_predictor(out, 'gru')
            np.testing.assert_allclose(p, loaded(frame.text.tolist()), atol=1e-7)
            self.assertEqual(p.shape[1], 28)

    def test_phobert_tiny_offline_full_finetune_and_reload(self):
        import torch
        from transformers import RobertaConfig, RobertaForSequenceClassification, BertTokenizer
        torch.set_num_threads(1)
        frame = fixture_frames().iloc[[0, 2, 27, 28]].copy()
        with tempfile.TemporaryDirectory(dir=study.ROOT) as tmp, patch.object(study, 'DEFAULT_OUTPUT', Path(tmp)):
            out = Path(tmp)
            source = out/'tiny_source'
            source.mkdir()
            vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'vui', 'buồn', 'bình', 'thường']
            (source/'vocab.txt').write_text('\n'.join(vocab), encoding='utf-8')
            BertTokenizer(str(source/'vocab.txt'), do_lower_case=False).save_pretrained(source)
            tiny = RobertaForSequenceClassification(RobertaConfig(vocab_size=len(vocab), hidden_size=8,
                num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, num_labels=28,
                max_position_embeddings=32, pad_token_id=0, bos_token_id=2, eos_token_id=3))
            original_embedding = tiny.roberta.embeddings.word_embeddings.weight.detach().clone()
            tiny.save_pretrained(source)
            cfg = study.study_config(study.read_json(study.ROOT/'configs/midterm.json'))
            cfg['phobert'].update(checkpoint=str(source), revision='offline', epochs=1, max_length=8, batch_size=2)
            folder = out/'models/phobert'
            folder.mkdir(parents=True)
            with patch('train_neural.segmented_texts', side_effect=lambda frames, _: {
                    s: f.normalized.tolist() for s, f in frames.items()}):
                p, meta = study.train_neural28('phobert', frame, frame, cfg, folder, out/'unused', allow_cpu=True)
            saved = RobertaForSequenceClassification.from_pretrained(folder/'pretrained', local_files_only=True)
            self.assertFalse(torch.equal(original_embedding, saved.roberta.embeddings.word_embeddings.weight))
            self.assertEqual(saved.classifier.out_proj.out_features, 28)
            self.assertEqual(meta['trainable_parameter_count'], meta['parameter_count'])
            write_json(folder/'run.json', dict(config=cfg, labels=study.LABELS28))
            fake_segmenter = SimpleNamespace(VnCoreNLP=lambda **_: SimpleNamespace(word_segment=lambda t: [t]))
            with patch.dict(sys.modules, {'py_vncorenlp': fake_segmenter}), \
                 patch.object(study, 'ensure_segmenter', return_value=out/'unused'):
                loaded = study.load_predictor(out, 'phobert')
                np.testing.assert_allclose(p, loaded(frame.text.tolist()), atol=1e-7)

    def test_neural_train_interface_has_no_test_argument(self):
        import inspect
        parameters = inspect.signature(study.train_neural28).parameters
        self.assertIn('train', parameters)
        self.assertIn('val', parameters)
        self.assertNotIn('test', parameters)

    def test_validation_demo_finds_all_four_categories_without_exporting_text(self):
        frame = fixture_frames().iloc[:4].copy()
        p = np.full((4,28), .1)
        p[0,27] = .8
        p[1,2] = .8
        p[2,[2,3]] = .8
        p[3,[2,27]] = .8
        with tempfile.TemporaryDirectory(dir=study.ROOT) as tmp:
            out = Path(tmp)
            study.save_evaluation(out, 'tfidf_lr', 'val', frame, p)
            write_json(out/'models/tfidf_lr/run.json', {})
            with patch.object(study, 'checked_study', return_value=(out, {}, {'val':frame})):
                cases = study.find_validation_cases(out, out)
            self.assertEqual(cases.available_count.tolist(), [1,1,1,1])
            self.assertNotIn('text', cases.columns)

    def test_test_evaluation_requires_all_models_and_prevents_future_training(self):
        with tempfile.TemporaryDirectory(dir=study.ROOT) as tmp:
            out = Path(tmp)
            with patch.object(study, 'checked_study', return_value=(out, {}, {})):
                with self.assertRaisesRegex(ValueError, 'Complete all three'):
                    study.evaluate_test(out, out)
                write_json(out/'test_lock.json', {'locked':True})
                with self.assertRaisesRegex(RuntimeError, 'locked'):
                    study.train_model(out, out, 'gru')
            self.assertFalse((out/'models').exists())

    def test_configuration_changes_cannot_silently_reuse_existing_study(self):
        with tempfile.TemporaryDirectory(dir=study.ROOT) as tmp, patch.object(study, 'DEFAULT_OUTPUT', Path(tmp)):
            out = Path(tmp)
            frame = fixture_frames()
            counts = pd.DataFrame([dict(split='train', raw_samples=len(frame), neutral_only=1, mixed_neutral=1)])
            write_json(out/'study.json', {'old_study':True})
            before = study.digest(out/'study.json')
            with patch.object(study, 'load_frames', return_value=({'train':frame}, frame.iloc[:0], counts)), \
                 patch.object(study, 'runtime', return_value={}):
                with self.assertRaises(FileExistsError):
                    study.init_study(out, out, study.ROOT/'configs/midterm.json', study.ROOT/'reports')
            self.assertEqual(study.digest(out/'study.json'), before)


if __name__ == '__main__':
    unittest.main()
