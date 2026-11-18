"""Build a small runnable Final code bundle; private data and weights stay separate."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from submission_package import write_public_member

ROOT = Path(__file__).resolve().parents[1]


def package(destination=None, root=ROOT):
    root = Path(root)
    names = ['common.py', 'train_neural.py', 'evaluate.py', 'neutral_28_labels.py',
             'import_equal20_results.py', 'q1_difficulty_common.py', 'submission_package.py',
             'final_threshold_core.py', 'final_study.py', 'final_inference.py', 'final_thresholds.py', 'package_final.py']
    paths = [root/'scripts'/name for name in names]
    paths += [root/name for name in ('configs/final_thresholds.json', 'configs/kaggle_equal_20ep.json',
              'docs/final_protocol.md', 'docs/final_execution.md', 'docs/final_label_tiers.csv',
              'notebooks/10_kaggle_final_validation.ipynb', 'tests/test_final_thresholds.py',
              'requirements-colab.txt', '.gitignore')]
    paths += [root/'splits'/f'{split}.csv' for split in ('train', 'val', 'test')]
    evidence = root/'reports/kaggle_equal20_2026-11-05'
    paths += [evidence/name for name in ('provenance.json', 'evidence/28/manifest.json',
              'evidence/28/phobert/run.json', 'evidence/28/phobert/val_predictions.npz')]
    target = Path(destination or root/'vigoemotions_final28_code.zip')
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        for path in paths:
            write_public_member(archive, path, root)
    return target


if __name__ == '__main__':
    print('Created:', package())
