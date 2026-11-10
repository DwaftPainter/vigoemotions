"""Build historical Kaggle entry notebooks from archived Colab workflows."""
import argparse
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
    candidates = []
    for path in Path('/kaggle/input').rglob(name):
        with path.open(encoding='utf-8-sig', newline='') as handle:
            header = next(csv.reader(handle), None)
        # ZIP code có splits/*.csv chỉ chứa ID, không phải dữ liệu nguồn.
        if path.parent.name == 'splits' and header == ['id']:
            continue
        if header != ['id', 'text', 'labels']:
            raise ValueError(f'{path}: schema phải là id,text,labels')
        candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(f'Cần đúng một {name} dữ liệu (id,text,labels) trong Input; tìm thấy {len(candidates)}. Kiểm tra Dataset đã attach.')
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
    source = ROOT / 'archive/notebooks' / f'{number}_colab_{suffix}.ipynb'
    notebook = deepcopy(json.loads(source.read_text(encoding='utf-8')))
    cells = notebook['cells']
    replace(cells[0], (
        f'# ViGoEmotions — Kaggle: {suffix.replace("_", " ")}\n'
        'Attach hai Dataset riêng tư: code và ba CSV. Bật GPU + Internet.\n'
        'Chạy từng cell từ trên xuống; nếu lỗi thì dừng. Hướng dẫn: archive/docs/kaggle_guide.md.\n'
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
    target = ROOT / 'archive/notebooks' / f'{number}_kaggle_{suffix}.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('Created:', target.name)


def build_neutral28(target=None):
    """Adapt notebook 06 I/O while preserving its isolated study and training cells."""
    source = ROOT / 'archive/notebooks/06_colab_neutral_28_labels.ipynb'
    notebook = json.loads(source.read_text(encoding='utf-8'))
    cells = notebook['cells']
    replace(cells[0], (
        '# ViGoEmotions — Kaggle: 28 nhãn, gồm Neutral\n\n'
        'So sánh TF-IDF + OvR LR, BiGRU và PhoBERT với baseline 27 nhãn.\n'
        'Attach hai Dataset Private: ZIP code 28 nhãn và ba CSV train/val/test.\n'
        'Trong Settings/Session options, bật GPU + Internet; không chọn TPU.\n'
        'Chạy từng cell từ trên xuống; dừng khi có lỗi. Hướng dẫn: archive/docs/neutral_28_labels.md.\n'
        'Artifact ở /kaggle/working/ViGoEmotions28/reports/neutral_28_labels/.\n'
        'Model hoàn tất được giữ; không hỗ trợ resume giữa epoch.\n'
    ))
    replace(cells[1], (
        '## 1. Nạp project từ Kaggle Input\n\n'
        'Attach Dataset chứa **vigoemotions_neutral28_colab.zip** mới, không dùng ZIP 04/05.\n'
        'Hỗ trợ cả ZIP và Dataset đã được Kaggle giải nén. Code tự tìm đường dẫn.\n'
        'Cell dưới chỉ chạy một lần trong workspace mới; không giải nén đè kết quả.\n'
        'Sau restart kernel, bỏ qua cell giải nén nếu project còn và chạy cell nối workspace.\n'
    ))
    bootstrap = BOOTSTRAP.replace('ViGoEmotions', 'ViGoEmotions28')
    bootstrap = bootstrap.replace('vigoemotions_colab.zip', 'vigoemotions_neutral28_colab.zip')
    bootstrap = bootstrap.replace('scripts/common.py', 'scripts/neutral_28_labels.py')
    replace(cells[2], bootstrap)
    replace(cells[8], DATA + "\nDATA = ROOT / 'data'\n")
    replace(cells[7], ''.join(cells[7]['source']).replace(
        'Upload riêng `train.csv`, `val.csv`, `test.csv` có quyền truy cập.',
        'Attach Dataset Private chứa `train.csv`, `val.csv`, `test.csv` có quyền truy cập.'))
    replace(cells[27], (
        '## 12. Lưu kết quả và checkpoint riêng tư\n\n'
        'Chạy sau inference/report. ZIP toàn bộ study gồm CSV, NPZ, config, checkpoint,\n'
        'tokenizer, vocabulary và segmenter; không chứa CSV/comment nguồn từ data/.\n'
        'Tải ZIP từ file browser/Output trong /kaggle/working trước khi kết thúc session.\n'
        'Giữ ZIP và notebook Private. Save & Run All sẽ chạy lại toàn bộ trong session mới.\n'
    ))
    for cell in cells:
        text = ''.join(cell['source']).replace('/content/', '/kaggle/working/')
        text = text.replace('  # Sửa nếu dùng project trên Drive.', '')
        text = text.replace('from google.colab import files\n', '')
        text = text.replace('files.download(str(archive_path))',
                            "print('Tải ZIP từ file browser/Output:', archive_path)")
        replace(cell, text)
        if cell['cell_type'] == 'code':
            ast.parse(text)
            cell['outputs'] = []
            cell['execution_count'] = None
    notebook.get('metadata', {}).pop('colab', None)
    target = Path(target) if target is not None else ROOT / 'archive/notebooks/06_kaggle_neutral_28_labels.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('Created:', target.name)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neutral28', action='store_true', help='Build/package only notebook 06 (28 labels)')
    args = parser.parse_args()
    if args.neutral28:
        subprocess.run([sys.executable, str(ROOT / 'scripts/package_neutral28.py')], check=True)
    else:
        build('04', 'neural_baselines')
        build('05', 'phobert_tuning')
        subprocess.run([sys.executable, str(ROOT / 'scripts/package_colab.py'), '--include-results'], check=True)
