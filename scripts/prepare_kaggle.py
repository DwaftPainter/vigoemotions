"""Build Kaggle entry notebooks from the maintained Colab workflows."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP = '''from pathlib import Path
import os, shutil, zipfile

INPUT = Path('/kaggle/input')
ROOT = Path('/kaggle/working/ViGoEmotions')
if ROOT.exists():
    raise FileExistsError(f'{ROOT} đã tồn tại. Không giải nén đè lên kết quả; chạy tiếp cell cần thiết hoặc dùng session mới.')

# Kaggle có thể giữ ZIP hoặc tự giải nén ZIP khi tạo Dataset.
archives = list(INPUT.rglob('vigoemotions_colab.zip'))
code_roots = [p.parent.parent for p in INPUT.rglob('scripts/common.py')]
if len(archives) == 1:
    with zipfile.ZipFile(archives[0]) as archive:
        for item in archive.infolist():
            if not (ROOT / item.filename).resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Unsafe ZIP path')
        ROOT.mkdir(parents=True)
        archive.extractall(ROOT)
elif not archives and len(code_roots) == 1:
    shutil.copytree(code_roots[0], ROOT)
else:
    raise RuntimeError('Hãy attach đúng một Dataset code chứa vigoemotions_colab.zip hoặc project đã giải nén.')
os.chdir(ROOT)
print('Project:', ROOT)
'''

DATA = '''from pathlib import Path
import csv, shutil

sources = {}
for name in ('train.csv', 'val.csv', 'test.csv'):
    candidates = list(Path('/kaggle/input').rglob(name))
    if len(candidates) != 1:
        raise RuntimeError(f'Cần đúng một {name} trong Input; tìm thấy {len(candidates)}. Kiểm tra Dataset đã attach.')
    with candidates[0].open(encoding='utf-8-sig', newline='') as handle:
        if next(csv.reader(handle)) != ['id', 'text', 'labels']:
            raise ValueError(f'{name}: schema phải là id,text,labels')
    sources[name] = candidates[0]
(ROOT / 'data').mkdir(exist_ok=True)
for name, source in sources.items():
    shutil.copy2(source, ROOT / 'data' / name)
    print('Đã nạp:', name)
'''

SETUP = '''%pip install -q -r requirements-colab.txt
import shutil, subprocess
if shutil.which('java') is None:
    subprocess.run(['apt-get', '-qq', 'update'], check=True)
    subprocess.run(['apt-get', '-qq', 'install', '-y', 'default-jre-headless'], check=True)
'''


def replace(cell, source):
    cell['source'] = source.splitlines(keepends=True)


def build(number, suffix):
    source = ROOT / 'notebooks' / f'{number}_colab_{suffix}.ipynb'
    notebook = deepcopy(json.loads(source.read_text(encoding='utf-8')))
    cells = notebook['cells']
    replace(cells[0], (
        f'# ViGoEmotions — Kaggle: {suffix.replace("_", " ")}\n'
        'Attach hai Dataset riêng tư: code và ba CSV. Bật GPU + Internet.\n'
        'Chạy từng cell từ trên xuống; nếu lỗi thì dừng. Hướng dẫn: docs/kaggle_guide.md.\n'
        'Cell đầu chỉ chạy một lần trong session mới. Kết quả lưu trong /kaggle/working.\n'
        'Đây là một lượt chạy mới; không dùng để resume sweep Colab khác môi trường.\n'
    ))
    replace(cells[1], BOOTSTRAP)
    replace(cells[2], SETUP)
    replace(cells[4], '## Nạp dữ liệu\nAttach Dataset chứa train.csv, val.csv, test.csv vào Input. Cell dưới copy CSV vào project để các script dùng chung đường dẫn.\n')
    replace(cells[5], DATA)
    for cell in cells:
        text = ''.join(cell['source'])
        text = text.replace('/content/', '/kaggle/working/')
        lines = []
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith('files.download('):
                argument = line.strip()[len('files.download('):-1]
                line = f"print('Output cần tải:', {argument})\n"
            lines.append(line)
        cell['source'] = lines
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            # Validate Python syntax apart from notebook magics.
            ast.parse(''.join(line for line in lines if not line.startswith(('%', '!'))))
    notebook.get('metadata', {}).pop('colab', None)
    target = ROOT / 'notebooks' / f'{number}_kaggle_{suffix}.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('Created:', target.name)


if __name__ == '__main__':
    build('04', 'neural_baselines')
    build('05', 'phobert_tuning')
    subprocess.run([sys.executable, str(ROOT / 'scripts/package_colab.py'), '--include-results'], check=True)
