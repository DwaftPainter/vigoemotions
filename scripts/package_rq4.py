"""Package the RQ4 Kaggle job from an allowlist of code and text-free RQ1 evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from package_final import ROOT, package as package_final
from submission_package import write_public_member

EXTRA = (
    'configs/rq4_noise_robustness.json',
    'docs/rq4_noise_robustness.md',
    'scripts/rq4_noise.py',
    'scripts/rq4_robustness.py',
    'scripts/package_rq4.py',
    'notebooks/11_kaggle_rq4_noise_robustness.ipynb',
    'reports/final28/thresholds.json',
    'reports/final28/validation_reproduction.json',
    'reports/final28/pre_test_lock.json',
    'reports/final28/test_attempt.json',
    'reports/final28/predictions/val_reproduced.npz',
    'reports/final28/predictions/val.npz',
    'reports/final28/predictions/test.npz',
)


def package(destination=None, root=ROOT):
    root = Path(root)
    target = Path(destination or root/'vigoemotions_rq4_code.zip').resolve()
    with TemporaryDirectory(prefix='.rq4-package-', dir=target.parent) as temporary:
        staged = package_final(Path(temporary)/target.name, root)
        with ZipFile(staged, 'a', ZIP_DEFLATED) as archive:
            for relative in EXTRA:
                write_public_member(archive, root/relative, root)
        staged.replace(target)
    return target


if __name__ == '__main__':
    print('Created:', package())
