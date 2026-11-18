"""Translate the preserved Midterm snapshot using unchanged tables and figures."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from docx import Document
from pypdf import PdfReader

from build_final_paper import fill_template, render_docx
from build_report import register_fonts, render_report

ROOT = Path(__file__).resolve().parents[1]
TABLE_KEYS = ('metrics27', 'metrics28', 'training', 'projection', 'neutral',
              'extremes', 'frequency', 'pairs', 'styles', 'sensitivity',
              'review', 'per_label27', 'per_label28')


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tables(markdown):
    return re.findall(r'(?:^\|[^\n]*\n)+', markdown, re.M)


def rows(table):
    return [[cell.strip() for cell in line.strip('|').split('|')]
            for line in table.strip().splitlines()
            if not re.fullmatch(r'[| :\-]+', line)]


def verify_translation(vietnamese, english):
    source, translated = tables(vietnamese), tables(english)
    if len(source) != len(TABLE_KEYS) or len(translated) != len(TABLE_KEYS):
        raise ValueError('Expected all 13 Midterm tables in both languages')
    for key, original, translation in zip(TABLE_KEYS, source, translated):
        a, b = rows(original), rows(translation)
        if key == 'review':
            if a[0] != b[0] or len(a) != len(b) or [r[:2] for r in a] != [r[:2] for r in b]:
                raise ValueError('Translated review table changed case identities')
        elif rows(original.replace('Không áp dụng', 'N/A')) != b:
            raise ValueError(f'Translated table differs from the Midterm source: {key}')
    image_paths = lambda text: re.findall(r'!\[.*?\]\((.*?)\)', text)
    if image_paths(vietnamese) != image_paths(english) or len(image_paths(english)) != 5:
        raise ValueError('Translated figure inventory differs from the Midterm source')


def translated_markdown(vietnamese):
    blocks = tables(vietnamese)
    if len(blocks) != len(TABLE_KEYS):
        raise ValueError('Midterm table inventory changed; update the English template')
    values = {key: block.strip().replace('Không áp dụng', 'N/A')
              for key, block in zip(TABLE_KEYS, blocks) if key != 'review'}
    template = (ROOT / 'docs/midterm_report_en_template.md').read_text()
    english = fill_template(template, values)
    verify_translation(vietnamese, english)
    return english


def build(destination=ROOT / 'reports'):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'reports/midterm_report_vi.md'
    vietnamese = source.read_text()
    images = [ROOT / 'reports' / p for p in re.findall(r'!\[.*?\]\((.*?)\)', vietnamese)]
    inputs = [source, ROOT / 'reports/midterm_report_vi.pdf', *images,
              ROOT / 'docs/midterm_report_en_template.md', Path(__file__).resolve(),
              ROOT / 'scripts/build_report.py', ROOT / 'scripts/build_final_paper.py',
              ROOT / 'requirements.txt']
    before = {str(p.relative_to(ROOT)): checksum(p) for p in inputs}
    english = translated_markdown(vietnamese)
    md, pdf, docx = [destination / f'midterm_report.{suffix}' for suffix in ('md', 'pdf', 'docx')]
    md.write_text(english, encoding='utf-8')
    register_fonts()
    render_report(english, destination=pdf, title='ViGoEmotions — Midterm Report', compact=True)
    render_docx(english, docx, ROOT / 'reports')
    word = Document(docx)
    word.core_properties.subject = 'ViGoEmotions Midterm: English translation'
    word.sections[0].footer.paragraphs[0].runs[0].text = 'ViGoEmotions | Midterm | ATI 2026 | '
    word.save(docx)
    for relative, sha in before.items():
        if checksum(ROOT / relative) != sha:
            raise ValueError(f'Midterm input changed during export: {relative}')
    result = dict(language='en', source_language='vi', scope='preserved Midterm snapshot',
                  numeric_tables_preserved=12, translated_review_cases=6, figures_preserved=5,
                  page_count=len(PdfReader(pdf).pages), training_or_inference_performed=False,
                  input_sha256=before,
                  generated_sha256={p.name: checksum(p) for p in (md, pdf, docx)})
    (destination / 'midterm_report_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports')
    result = build(parser.parse_args().output_dir)
    print(json.dumps({k: result[k] for k in ('language', 'page_count', 'numeric_tables_preserved',
                                          'translated_review_cases', 'figures_preserved')}, indent=2))
