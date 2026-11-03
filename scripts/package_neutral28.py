"""Package notebook 06 + code + real 27-label reference artifacts; no private data."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def package():
    paths = [ROOT/name for name in (
        'scripts/common.py', 'scripts/train_neural.py', 'scripts/evaluate.py',
        'scripts/neutral_28_labels.py', 'tests/test_neutral_28_labels.py',
        'configs/midterm.json', 'notebooks/06_colab_neutral_28_labels.ipynb',
        'docs/neutral_28_labels.md')]
    paths.extend((ROOT/'splits').glob('*.csv'))  # IDs only, not dataset CSVs.
    for name in ('tfidf_lr', 'gru', 'phobert'):
        for relative in (f'reports/{name}_run.json', f'reports/tables/{name}_metrics.csv',
                         f'reports/predictions/{name}_val.npz'):
            path = ROOT/relative
            if path.is_file():
                paths.append(path)
    target = ROOT/'vigoemotions_neutral28_colab.zip'
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print('Created:', target)
    print('No raw comments/CSV, trained 28-label model, vocabulary or token included.')


if __name__ == '__main__':
    package()
