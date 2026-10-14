"""Package code and source documents only; never include restricted data."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
files = []
for folder in ('scripts', 'configs', 'tests', 'docs', 'notebooks'):
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts)
files.extend(ROOT/name for name in ('README.md','requirements-colab.txt','requirements.txt',
                                    '02-vigoemotions-vietnamese-emotion.pdf','student-guide.pdf'))
with ZipFile(ROOT/'vigoemotions_colab.zip','w',ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path,path.relative_to(ROOT))
print('Created vigoemotions_colab.zip: code/notebooks/docs only, no raw data or vocab.')
