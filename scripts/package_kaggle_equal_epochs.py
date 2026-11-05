"""Package the canonical equal-20-epoch Kaggle workflow; never include experiment data/results."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from submission_package import write_public_member

ROOT = Path(__file__).resolve().parents[1]


def package():
    paths = [ROOT/'scripts'/name for name in (
        'common.py', 'evaluate.py', 'train_tfidf.py', 'train_neural.py',
        'neutral_28_labels.py', 'kaggle_equal_epochs.py', 'submission_package.py')]
    paths += [ROOT/name for name in ('configs/kaggle_equal_20ep.json',
        'notebooks/08_kaggle_equal_20ep_27_labels.ipynb',
        'notebooks/09_kaggle_equal_20ep_28_labels.ipynb', 'tests/test_kaggle_equal_epochs.py',
        'docs/kaggle_equal_epochs.md', 'requirements-colab.txt')]
    paths += sorted((ROOT/'splits').glob('*.csv'))
    target = ROOT/'vigoemotions_kaggle_equal_20ep.zip'
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        for path in paths:
            write_public_member(archive, path, ROOT)
    print('Created:', target)
    print('Code/config/split IDs only. No historical scores, raw data, weights or vocabulary.')
    return target


if __name__ == '__main__':
    package()
