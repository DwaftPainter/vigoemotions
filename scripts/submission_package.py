"""Write shareable code archives without notebook outputs or private study artifacts."""
from copy import deepcopy
import csv
import json


def sanitized_notebook(notebook):
    clean = deepcopy(notebook)
    clean['metadata'] = {key: value for key, value in clean.get('metadata', {}).items()
                         if key in ('kernelspec', 'language_info')}
    for cell in clean['cells']:
        cell['metadata'] = {}
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    return clean


def is_public_report(path, root):
    parts = path.relative_to(root).parts
    if any(part in {'neutral_28_labels', 'resources', 'tokenizer', 'pretrained', 'checkpoints'} for part in parts):
        return False
    if path.suffix not in {'.csv', '.json', '.md', '.pdf', '.png', '.npz'}:
        return False
    if 'vocab' in path.name.lower() or path.name.startswith('prediction_example'):
        return False
    if path.suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            columns = next(csv.reader(stream), [])
        if {'text', 'normalized', 'comment', 'raw_text', 'input_text'} & set(columns):
            return False
    return True


def write_public_member(archive, path, root):
    if path.suffix == '.ipynb':
        content = sanitized_notebook(json.loads(path.read_text(encoding='utf-8')))
        archive.writestr(path.relative_to(root).as_posix(), json.dumps(content, ensure_ascii=False, indent=1)+'\n')
    else:
        archive.write(path, path.relative_to(root).as_posix())
