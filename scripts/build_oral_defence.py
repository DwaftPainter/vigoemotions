"""Build five study notes from retained evidence; never train or run model inference."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape, quoteattr

import matplotlib
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                               SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / 'docs/oral_defence'
METHODS = ('fixed', 'global', 'per_label')
NAMES = {'fixed': 'Fixed 0.5', 'global': 'Global', 'per_label': 'Per-label'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(root=ROOT):
    manifest = json.loads((root/'docs/oral_defence/source_manifest.json').read_text())
    sources = manifest['retained_original_sha256']
    for relative, expected in sources.items():
        path = root / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Retained source differs from the research branch: {relative}')
    return sources


def table(headers, rows):
    lines = ['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |']
    lines.extend('| '+' | '.join(map(str, row))+' |' for row in rows)
    return '\n'.join(lines)


def substitutions(root=ROOT):
    def csv_rows(relative):
        with (root/relative).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    def metric(name, scope='all28'):
        return {r['strategy']: r for r in csv_rows(f'reports/final28/tables/{name}.csv') if r['scope'] == scope}
    def number(value, digits=6):
        return f'{float(value):.{digits}f}' if value != '' else 'N/A'
    val, test, emotion = metric('val_comparison'), metric('test_comparison'), metric('test_comparison', 'emotion27')
    thresholds = json.loads((root/'reports/final28/thresholds.json').read_text())
    fixed = float(test['fixed']['macro_f1'])
    values = {f'test_{key}_macro': number(test[key]['macro_f1']) for key in METHODS}
    values.update(global_threshold=number(thresholds['global_threshold'], 2),
                  neutral_threshold=number(thresholds['per_label_thresholds'][27], 2),
                  val_per_label_macro=number(val['per_label']['macro_f1']),
                  global_gain_pp=f"{100*(float(test['global']['macro_f1'])-fixed):+.4f}",
                  per_label_gain_pp=f"{100*(float(test['per_label']['macro_f1'])-fixed):+.4f}")
    values['primary_table'] = table(['Decoder', 'Val Macro', 'Test Macro', 'Test Micro', 'Hamming', 'Δ Macro pp'], [
        [NAMES[s], number(val[s]['macro_f1']), number(test[s]['macro_f1']), number(test[s]['micro_f1']),
         number(test[s]['hamming_loss']), f"{100*(float(test[s]['macro_f1'])-fixed):+.4f}"] for s in METHODS])
    values['emotion27_table'] = table(['Decoder', 'Test Macro27', 'Test Micro27', 'Hamming27'], [
        [NAMES[s], *[number(emotion[s][key]) for key in ('macro_f1', 'micro_f1', 'hamming_loss')]] for s in METHODS])
    gaps = {r['strategy']: r for r in csv_rows('reports/final28/tables/generalization_gap.csv')
            if r['scope'] == 'all28' and r['metric'] == 'macro_f1'}
    values['gap_table'] = table(['Decoder', 'Val−test Macro pp'], [
        [NAMES[s], f"{100*float(gaps[s]['gap']):+.4f}"] for s in METHODS])
    tiers = {(r['tier'], r['strategy']): r for r in csv_rows('reports/final28/tables/test_frequency_tiers.csv')}
    values['tier_table'] = table(['Tier', 'Fixed Macro', 'Global Macro', 'Per-label Macro', 'PL Δ pp'], [
        [tier, *[number(tiers[tier, s]['macro_f1']) for s in METHODS],
         f"{100*float(tiers[tier, 'per_label']['delta_macro_f1_vs_fixed']):+.4f}"] for tier in ('head', 'mid', 'tail')])
    cards = {(r['cardinality'], r['strategy']): r for r in csv_rows('reports/final28/tables/test_cardinality.csv') if r['scope'] == 'all28'}
    values['cardinality_table'] = table(['Card. all28', 'N', 'Fixed Macro', 'Global Macro', 'Per-label Macro'], [
        [group, cards[group, 'fixed']['n'], *[number(cards[group, s]['macro_f1']) for s in METHODS]] for group in ('0', '1', '2', '3', '>=4')])
    neutrals = {r['strategy']: r for r in csv_rows('reports/final28/tables/test_neutral.csv')}
    values['neutral_table'] = table(['Decoder', 'Cutoff', 'TP', 'FP', 'FN', 'Neutral F1'], [
        [NAMES[s], number(neutrals[s]['threshold'], 2), *[neutrals[s][key] for key in ('tp', 'fp', 'fn')], number(neutrals[s]['f1'])] for s in METHODS])
    profile = csv_rows('reports/final28/tables/threshold_profile.csv')
    values['threshold_table'] = table(['Nhãn', 'Cutoff', 'Val positives', 'Train positives', 'Tier'], [
        [r['label'], number(r['threshold'], 2), r['validation_support'], r['train_count'], r['tier']] for r in profile])
    baseline = csv_rows('reports/kaggle_equal20_2026-11-05/model_comparison.csv')
    names = {'tfidf_lr': 'TF-IDF + LR', 'gru': 'BiGRU', 'phobert': 'PhoBERT'}
    values['midterm_table'] = table(['Mô hình', 'Outputs', 'Val Macro-F1', 'Best epoch'], [
        [names[r['model']], r['label_count'], number(r['macro_f1']),
         str(int(float(r['best_epoch']))) if r['best_epoch'] and r['best_epoch'].lower() != 'nan' else 'N/A'] for r in baseline])
    locations = [('common.py', 'scores', 'Macro/Micro, Hamming, LRAP'),
                 ('train_tfidf.py', 'main', 'Fit TF-IDF + OvR LR'),
                 ('train_neural.py', 'BiGRU', 'Embedding, GRU hai hướng, linear head'),
                 ('neutral_28_labels.py', 'train_neural28', 'Full fine-tuning và checkpoint epoch 15'),
                 ('final_threshold_core.py', 'fit_thresholds', 'Grid, tie-break và fallback'),
                 ('final_threshold_core.py', 'decode', 'Scalar/vector cutoffs, so sánh >='),
                 ('final_study.py', 'verify_lock', 'Kiểm source và lock bindings'),
                 ('final_inference.py', 'predict_checkpoint', 'Saved model/tokenizer, eval và sigmoid')]
    code = []
    for filename, name, purpose in locations:
        lines = (root/'scripts'/filename).read_text().splitlines()
        line = next(i for i, content in enumerate(lines, 1) if re.match(rf'\s*(?:def|class) {name}\b', content))
        code.append([f'[{filename}](../../scripts/{filename})', f'{name}, dòng {line}', purpose])
    values['code_map'] = table(['File', 'Vị trí', 'Ý cần hiểu'], code)
    return values


def inline(text):
    text = escape(text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: f'<link href={quoteattr(m[2])} color="#20566b">{m[1]}</link>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return re.sub(r'`([^`]+)`', r'<font color="#20566b">\1</font>', text)


def render_pdf(markdown, destination):
    fonts = Path(matplotlib.get_data_path())/'fonts/ttf'
    for name, file in [('Study', 'DejaVuSans.ttf'), ('Study-Bold', 'DejaVuSans-Bold.ttf'), ('StudyMono', 'DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fonts/file)))
    pdfmetrics.registerFontFamily('Study', normal='Study', bold='Study-Bold', italic='Study', boldItalic='Study-Bold')
    body = ParagraphStyle('body', fontName='Study', fontSize=9.2, leading=12.5, spaceAfter=4)
    heading = ParagraphStyle('heading', parent=body, fontName='Study-Bold', fontSize=12.5, leading=17,
                             spaceBefore=10, spaceAfter=5, keepWithNext=True)
    subheading = ParagraphStyle('subheading', parent=heading, fontSize=10.2, leading=14, spaceBefore=8)
    title = ParagraphStyle('title', parent=heading, fontSize=17, leading=23, alignment=TA_CENTER, spaceAfter=14)
    cell = ParagraphStyle('cell', parent=body, fontSize=7.2, leading=10, spaceAfter=0)
    code = ParagraphStyle('code', fontName='StudyMono', fontSize=7.4, leading=11, spaceAfter=8)
    story, lines, i, width = [], markdown.splitlines(), 0, A4[0]-92
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if not line:
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak()); continue
        if line.startswith('```'):
            block = []
            while i < len(lines) and not lines[i].startswith('```'):
                block.append(lines[i]); i += 1
            i += 1; story.append(Preformatted('\n'.join(block), code)); continue
        if line.startswith('|'):
            raw = [line]
            while i < len(lines) and lines[i].strip().startswith('|'):
                raw.append(lines[i].strip()); i += 1
            rows = [[c.strip() for c in r.strip('|').split('|')] for r in raw if not re.fullmatch(r'[| :\-]+', r)]
            weights = [1.3]+[1]*(len(rows[0])-1) if len(rows[0]) > 2 else [1, 2.2]
            grid = Table([[Paragraph(('<b>'+inline(c)+'</b>') if row == 0 else inline(c), cell)
                           for c in values] for row, values in enumerate(rows)],
                         colWidths=[width*w/sum(weights) for w in weights], repeatRows=1, hAlign='LEFT')
            grid.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6eff2')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f8fa')]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), .5, colors.HexColor('#53717d')),
                ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
            story.extend([grid, Spacer(1, 8)]); continue
        picture = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
        if picture:
            caption, relative = picture.groups(); path = NOTES / relative
            w, h = ImageReader(str(path)).getSize(); scale = min(width/w, 280/h)
            story.append(KeepTogether([Image(str(path), w*scale, h*scale), Paragraph(inline(caption), cell), Spacer(1, 8)])); continue
        level = len(line)-len(line.lstrip('#'))
        if level:
            story.append(Paragraph(inline(line[level:].strip()), title if level == 1 else heading if level == 2 else subheading)); continue
        paragraph = [line]
        while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#', '|', '- ', '```', '![', '<!--')):
            paragraph.append(lines[i].strip()); i += 1
        text = ' '.join(paragraph)
        story.append(Paragraph(inline(text[2:] if text.startswith('- ') else text), body,
                               bulletText='•' if text.startswith('- ') else None))
    def footer(canvas, doc):
        canvas.saveState(); canvas.setFillColor(colors.white)
        canvas.rect(0, 0, *A4, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor('#53646d')); canvas.setFont('Study', 7.3)
        canvas.drawString(46, 25, 'ViGoEmotions | Ôn oral defence | 28 outputs')
        canvas.drawRightString(A4[0]-46, 25, str(doc.page)); canvas.restoreState()
    SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=46, rightMargin=46, topMargin=40,
                      bottomMargin=47, title=markdown.splitlines()[0].lstrip('# ')).build(story, onFirstPage=footer, onLaterPages=footer)


def build(destination=NOTES):
    before = verify_sources(); values = substitutions()
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    templates = sorted((NOTES/'templates').glob('*.md'))
    if len(templates) != 5:
        raise ValueError('Expected the five study-note templates')
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [*templates, Path(__file__).resolve(),
              ROOT/'requirements-study.txt', NOTES/'source_manifest.json', NOTES/'cleanup_manifest.json']}
    documents = []
    for template in templates:
        text = re.sub(r'\{\{([a-z][a-z0-9_]*)\}\}', lambda m: values[m[1]], template.read_text())
        if '{{' in text or '}}' in text:
            raise ValueError(f'Unresolved note template: {template.name}')
        md, pdf = destination/template.name, destination/template.with_suffix('.pdf').name
        md.write_text(text, encoding='utf-8'); render_pdf(text, pdf)
        reader = PdfReader(pdf); extracted = '\n'.join(p.extract_text() or '' for p in reader.pages)
        if len(extracted) < 1000 or '\ufffd' in extracted or 'ViGoEmotions' not in extracted:
            raise ValueError(f'Invalid PDF text export: {pdf.name}')
        documents.append(dict(markdown=md.name, pdf=pdf.name, pages=len(reader.pages),
                              markdown_sha256=sha(md), pdf_sha256=sha(pdf)))
    if before != verify_sources() or any(sha(ROOT/p) != value for p, value in inputs.items()):
        raise ValueError('Source changed while exporting study notes')
    result = dict(base_commit=json.loads((NOTES/'source_manifest.json').read_text())['base_commit'],
                  documents=documents, model_inference_performed=False, private_text_read=False,
                  retained_original_sha256=before, builder_inputs_sha256=inputs,
                  packages={name: importlib.metadata.version(name) for name in ('reportlab', 'matplotlib', 'pypdf')})
    (destination/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=NOTES)
    result = build(parser.parse_args().output_dir)
    print(json.dumps({'documents': [{k: item[k] for k in ('pdf', 'pages')} for item in result['documents']],
                      'model_inference_performed': False}, indent=2))
