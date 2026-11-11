"""Frozen PhoBERT-27: restore -> verify -> reproduce-val -> fit -> lock -> infer-test."""
import argparse
import json
from pathlib import Path

from common import ROOT
from final_study import Study
from final_inference import restore, reproduce_validation, infer_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'restore', 'reproduce-val', 'fit', 'lock', 'infer-test', 'report'])
    parser.add_argument('--config', type=Path, default=ROOT/'configs/final_thresholds.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reports/final')
    parser.add_argument('--private-dir', type=Path, default=ROOT/'data/final')
    parser.add_argument('--archive', type=Path, help='Private equal20 27-label ZIP; its SHA-256 must match the protocol')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--resume-failed', action='store_true', help='Resume only a failed technical test attempt with the identical lock')
    args = parser.parse_args()
    if args.resume_failed and args.command != 'infer-test':
        parser.error('--resume-failed is only valid for infer-test')
    try:
        study = Study(config=args.config, output=args.output_dir, data_dir=args.data_dir)
        if args.command == 'verify':
            reference = study.reference()
            result = dict(status='source_evidence_verified', validation_shape=list(reference['probability_vector'].shape),
                          frequency_tiers=study.tiers(), validation_inference_verified=False, test_evaluated=False)
        elif args.command == 'restore':
            archive = args.archive or ROOT/study.config['model']['archive_path']
            result = restore(study, archive, args.private_dir)
        elif args.command == 'reproduce-val':
            result = reproduce_validation(study, args.private_dir, args.device)
        elif args.command == 'fit':
            result = study.fit()
        elif args.command == 'lock':
            result = study.lock()
        elif args.command == 'report':
            result = study.report()
        else:
            result = infer_test(study, args.private_dir, args.device, args.resume_failed)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(1, f'{type(error).__name__}: {error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
