"""Restore private frozen assets and run inference without any training calls."""
import importlib.metadata
from pathlib import Path, PurePosixPath
import platform
import shutil
from zipfile import ZipFile

import numpy as np
import pandas as pd

from neutral_28_labels import LABELS28 as LABELS, targets28 as targets
from final_study import create_json, digest, read_json
from final_threshold_core import decode, read_predictions, validate_arrays, write_evaluation

REQUIRED_ASSETS = {'pretrained/model.safetensors', 'pretrained/config.json', 'tokenizer/vocab.txt', 'tokenizer/bpe.codes',
                   'tokenizer/tokenizer_config.json', 'tokenizer/special_tokens_map.json', 'vncorenlp/VnCoreNLP-1.2.jar',
                   'vncorenlp/models/wordsegmenter/wordsegmenter.rdr', 'vncorenlp/models/wordsegmenter/vi-vocab'}


def runtime_versions():
    return dict(python=platform.python_version(), **{name: importlib.metadata.version(name)
                for name in ('torch', 'transformers', 'py-vncorenlp', 'numpy', 'safetensors')})


def private_directory(study, private):
    private = Path(private).resolve()
    if private == study.root or (study.root in private.parents and not private.is_relative_to(study.root/'data')):
        raise ValueError('Weights and segmentation assets must stay in ignored data/ or outside the repository')
    if private == study.output or private.is_relative_to(study.output) or study.output.is_relative_to(private):
        raise ValueError('Private model assets and public results must be separate')
    return private


def restore(study, archive_path, private):
    private = private_directory(study, private)
    cfg = study.config['model']
    if digest(archive_path) != cfg['archive_sha256']:
        raise ValueError('Private model archive checksum mismatch')
    if (private/'assets.json').exists():
        return verify_assets(study, private)
    if private.exists() and any(private.iterdir()):
        raise FileExistsError('Refusing to overwrite a partial/existing private asset directory')
    private.mkdir(parents=True, exist_ok=True)
    files = {}
    with ZipFile(archive_path) as archive:
        selected = []
        for prefix, destination in [(cfg['model_prefix'], 'pretrained'), (cfg['tokenizer_prefix'], 'tokenizer')]:
            selected += [(name, destination+'/'+name[len(prefix):]) for name in archive.namelist()
                         if name.startswith(prefix) and not name.endswith('/')]
        prefix = 'resources/vncorenlp/'
        selected += [(name, 'vncorenlp/'+name[len(prefix):]) for name in archive.namelist()
                     if name.startswith(prefix) and not name.endswith('/')]
        if len(set(archive.namelist())) != len(archive.namelist()):
            raise ValueError('Duplicate archive members')
        for source, relative in selected:
            parts = PurePosixPath(relative)
            if parts.is_absolute() or '..' in parts.parts:
                raise ValueError('Unsafe private asset path')
            dest = private/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(source) as src, dest.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            files[relative] = digest(dest)
    if not REQUIRED_ASSETS <= set(files) or files[cfg['checkpoint_local_path']] != cfg['checkpoint_sha256']:
        raise ValueError('Missing segmentation assets or wrong frozen checkpoint')
    create_json(private/'assets.json', dict(archive_sha256=cfg['archive_sha256'], files_sha256=files))
    return verify_assets(study, private)


def verify_assets(study, private):
    private = private_directory(study, private)
    metadata = read_json(private/'assets.json')
    if not REQUIRED_ASSETS <= set(metadata['files_sha256']):
        raise ValueError('Required frozen inference assets are missing')
    if metadata['archive_sha256'] != study.config['model']['archive_sha256']:
        raise ValueError('Restored asset archive mismatch')
    for relative, checksum in metadata['files_sha256'].items():
        path = (private/relative).resolve()
        if not path.is_relative_to(private) or digest(path) != checksum:
            raise ValueError('Frozen checkpoint or segmentation assets changed')
    cfg = study.config['model']
    if metadata['files_sha256'].get(cfg['checkpoint_local_path']) != cfg['checkpoint_sha256']:
        raise ValueError('Restored checkpoint differs from protocol')
    if metadata['files_sha256']['pretrained/config.json'] != cfg['checkpoint_config_sha256']:
        raise ValueError('Saved model configuration differs from the archived checkpoint')
    model_config = read_json(private/'pretrained/config.json')
    if [model_config['id2label'].get(str(c)) for c in range(28)] != LABELS:
        raise ValueError('Saved model must contain the canonical 28-output label mapping')
    return metadata


def predict_checkpoint(study, frame, private, device_name=None):
    """Reuse the original segmentation and prediction helpers, with lazy GPU imports."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from train_neural import segmented_texts, predict
        importlib.import_module('py_vncorenlp')
    except ImportError as error:
        raise RuntimeError('Inference requires torch, transformers==4.57.3, py-vncorenlp==0.1.4 and Java; see docs/final_execution.md') from error
    if shutil.which('java') is None:
        raise RuntimeError('Java is required for the original VnCoreNLP segmentation')
    device = torch.device(device_name or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    torch.manual_seed(study.config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(study.config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    frames = {'train': pd.DataFrame({'normalized': pd.Series([], dtype=str)}), 'val': frame}
    text = segmented_texts(frames, Path(private)/'vncorenlp')['val']
    prep = study.config['preprocessing']
    tokenizer = AutoTokenizer.from_pretrained(Path(private)/'tokenizer', use_fast=False, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        Path(private)/'pretrained', local_files_only=True)
    if model.config.num_labels != 28 or [model.config.id2label[c] for c in range(28)] != LABELS:
        raise ValueError('Restored PhoBERT must retain the canonical 28-output head')
    model.requires_grad_(False)
    model.to(device)
    encoded = tokenizer(text, padding='max_length', truncation=True, max_length=prep['max_length'], return_tensors='pt')
    loader = DataLoader(TensorDataset(encoded['input_ids'], encoded['attention_mask'], torch.zeros((len(text), 28))),
                        batch_size=prep['inference_batch_size'], shuffle=False, num_workers=0)
    values = predict(model, 'phobert', loader, device)
    metadata = dict(device=str(device), packages=runtime_versions(),
                    segmentation='train_neural.segmented_texts', prediction='train_neural.predict')
    return values, metadata


def save_probabilities(path, frame, probabilities):
    validate_arrays(targets(frame), probabilities)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, sample_id=frame.id.to_numpy(dtype=str), labels=np.array(LABELS),
                            true_label_vector=targets(frame), probability_vector=probabilities,
                            label_cardinality=targets(frame).sum(axis=1), emotion_cardinality=targets(frame)[:, :27].sum(axis=1))


def reproduce_validation(study, private, device=None):
    if any((study.output/name).exists() for name in ('validation_reproduction.json', 'thresholds.json', 'pre_test_lock.json', 'test_attempt.json')):
        raise FileExistsError('Validation reproduction must use a fresh study output')
    assets = verify_assets(study, private)
    sources = study.sources()
    reference, frame = study.reference(), study.frame('val')
    actual, runtime = predict_checkpoint(study, frame, private, device)
    validate_arrays(reference['true_label_vector'], actual)
    expected = reference['probability_vector']
    if not np.allclose(actual, expected, rtol=0, atol=1e-4) or not np.array_equal(decode(actual, .5), decode(expected, .5)):
        raise ValueError('Frozen model does not reproduce full validation scores/decisions; no thresholds or test gate created')
    if study.sources() != sources or verify_assets(study, private) != assets:
        raise ValueError('Inputs changed during validation inference')
    path = study.output/'predictions/val_reproduced.npz'
    save_probabilities(path, frame, actual)
    report = dict(status='verified', sources_sha256=sources, prediction_sha256=digest(path),
                  checkpoint_sha256=study.config['model']['checkpoint_sha256'], assets=assets,
                  max_absolute_error=float(np.abs(np.asarray(actual, dtype=float)-expected).max()),
                  absolute_tolerance=1e-4, relative_tolerance=0, fixed_decisions_identical=True, runtime=runtime)
    create_json(study.output/'validation_reproduction.json', report)
    return report


def replace_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    create_json(temporary, value)
    temporary.replace(path)


def infer_test(study, private, device=None, resume_failed=False):
    study.verify_lock()  # This must run before reading any held-out text or targets.
    assets = verify_assets(study, private)
    reproduction = study.reproduction()
    if reproduction['assets'] != assets:
        raise ValueError('Test inference must use the validation-verified assets')
    runtime_reference = reproduction['runtime']
    if runtime_versions() != runtime_reference['packages'] or (device is not None and device != runtime_reference['device']):
        raise ValueError('Use the same runtime packages/device as verified validation inference')
    attempt_path = study.output/'test_attempt.json'
    lock_sha = digest(study.output/'pre_test_lock.json')
    if attempt_path.exists():
        previous = read_json(attempt_path)
        if not resume_failed or previous['status'] != 'failed' or previous['lock_sha256'] != lock_sha:
            raise FileExistsError('Test already attempted; only an unchanged failed technical attempt can resume')
        attempt = dict(previous, status='running', attempts=previous['attempts']+1)
    else:
        attempt = dict(status='running', lock_sha256=lock_sha, attempts=1)
    running = study.output/'test_running.lock'
    with running.open('x') as stream:
        stream.write(lock_sha)
    try:
        if attempt_path.exists(): replace_json(attempt_path, attempt)
        else: create_json(attempt_path, attempt)
        raw_path = study.output/'predictions/test_probabilities.npz'
        if raw_path.exists():
            if digest(raw_path) != attempt.get('raw_probabilities_sha256'):
                raise ValueError('Partial/unbound test probabilities must be investigated before resuming')
            values = read_predictions(raw_path, study.ids('test'))
        else:
            frame = study.frame('test')
            probabilities, runtime = predict_checkpoint(study, frame, private, device)
            if runtime != runtime_reference:
                raise ValueError('Runtime changed since verified validation inference')
            study.verify_lock()
            if verify_assets(study, private) != assets:
                raise ValueError('Model assets changed during test inference')
            save_probabilities(raw_path, frame, probabilities)  # Persist continuous scores before decoding.
            attempt.update(raw_probabilities_sha256=digest(raw_path), runtime=runtime)
            replace_json(attempt_path, attempt)
            values = read_predictions(raw_path, study.ids('test'))
        rows = write_evaluation(study.output, 'test', values, study.fitted(), study.tiers())
        study.write_gap()
        attempt.update(status='completed', predictions_sha256=digest(study.output/'predictions/test.npz'))
        replace_json(attempt_path, attempt)
        return rows
    except Exception as error:
        attempt.update(status='failed', error_type=type(error).__name__)
        replace_json(attempt_path, attempt)
        raise
    finally:
        running.unlink()
