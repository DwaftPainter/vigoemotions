"""Package code and source documents only; never include restricted data."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--include-results', action='store_true',
                    help='Include derived reports to compare tuning with existing baselines')
args = parser.parse_args()
files = []
for folder in ('scripts', 'configs', 'tests', 'docs', 'notebooks') + (('reports',) if args.include_results else ()):
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts)
files.extend(ROOT/name for name in ('README.md','README_PRE_MIDTERM.md',
                                    'requirements-colab.txt','requirements.txt'))
with ZipFile(ROOT/'vigoemotions_colab.zip','w',ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path,path.relative_to(ROOT))
print('Created vigoemotions_colab.zip; raw data, checkpoints and vocab excluded.')
