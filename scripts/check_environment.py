"""Fail early with an actionable message when the notebook kernel is incompatible."""
import importlib
from importlib.metadata import version, PackageNotFoundError
import sys


def check_environment():
    print('Python kernel:', sys.executable)
    try:
        for name in ('numpy', 'scipy', 'scikit-learn'):
            print(f'{name}: {version(name)}')
        # Import the exact dependency path used by the EDA and baseline cells.
        for name in ('numpy', 'scipy.sparse', 'sklearn.metrics'):
            importlib.import_module(name)
    except (ImportError, AttributeError, ValueError, PackageNotFoundError) as exc:
        raise RuntimeError(
            'Kernel hiện tại thiếu thư viện hoặc có dependency không tương thích. '
            'Chọn kernel "Python (ViGoEmotions)" dùng ViGoEmotions/.venv/bin/python, '
            'sau đó Restart Kernel và Run All. '
            f'Lỗi gốc: {type(exc).__name__}: {exc}'
        ) from None
    print('NumPy / SciPy / scikit-learn imports: OK')


if __name__ == '__main__':
    check_environment()
