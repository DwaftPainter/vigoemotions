"""Build the English Final paper from verified public artifacts; no neural imports."""
import argparse
import importlib.metadata
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.shared import Mm, Pt, RGBColor
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                               SimpleDocTemplate, Spacer, Table, TableStyle)

from build_final_figures import checksum, read_json, verify_inputs
from build_report import register_fonts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/final28'
STRATEGIES = ('fixed', 'global', 'per_label')
NAMES = {'fixed': 'Fixed 0.5', 'global': 'Global', 'per_label': 'Per-label'}


def markdown_table(frame):
    def value(v):
        if pd.isna(v):
            return 'N/A'
        return f'{v:.6f}' if isinstance(v, (float, np.floating)) else str(v)
    rows = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows += ['| ' + ' | '.join(value(v) for v in row) + ' |'
             for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def comparison_table(validation, test):
    validation = validation[validation.scope == 'all28'].set_index('strategy')
    test = test[test.scope == 'all28'].set_index('strategy')
    return markdown_table(pd.DataFrame([
        {'Decoder': NAMES[s], 'Val Macro-F1': validation.loc[s, 'macro_f1'],
         'Test Macro-F1': test.loc[s, 'macro_f1'], 'Test Micro-F1': test.loc[s, 'micro_f1'],
         'Hamming Loss': test.loc[s, 'hamming_loss'], 'LRAP': test.loc[s, 'lrap'],
         'Δ Macro (pp)': f"{100*(test.loc[s, 'macro_f1']-test.loc['fixed', 'macro_f1']):+.4f}"}
        for s in STRATEGIES]))


def fill_template(template, values):
    pattern = r'\{\{([a-z][a-z0-9_]*)\}\}'
    fields = set(re.findall(pattern, template))
    if fields - values.keys():
        raise ValueError(f'Unresolved manuscript template fields: {sorted(fields-values.keys())}')
    result = re.sub(pattern, lambda match: str(values[match[1]]), template)
    if '{{' in result or '}}' in result:
        raise ValueError('Unresolved manuscript template syntax')
    return result


def evidence():
    verify_inputs(OUT)
    reproduction = read_json(OUT / 'validation_reproduction.json')
    inputs = {str(OUT / p): sha for p, sha in read_json(OUT / 'execution_verification.json')['imported_members_sha256'].items()}
    inputs[str(OUT / 'execution_verification.json')] = checksum(OUT / 'execution_verification.json')
    for name, sha in reproduction['sources_sha256'].items():
        if name.startswith('private_source/'):
            continue  # Manuscript builds use derived evidence, never private CSVs.
        path = ROOT / ('configs/final_thresholds.json' if name == 'final_config' else name)
        if checksum(path) != sha:
            raise ValueError(f'Frozen source differs from execution evidence: {name}')
        inputs[str(path)] = sha
    figure_meta = OUT / 'analysis/figure_provenance.json'
    figures = read_json(figure_meta)
    for relative, sha in figures['generated_sha256'].items():
        path = OUT / relative
        if checksum(path) != sha:
            raise ValueError(f'Figure/analysis hash mismatch: {relative}')
        inputs[str(path)] = sha
    if checksum(ROOT / figures['builder']) != figures['builder_sha256']:
        raise ValueError('Figure builder changed since verified figure generation')
    inputs[str(figure_meta)] = checksum(figure_meta)
    audit = OUT / 'human_audit'
    provenance = read_json(audit / 'ai_assisted_provenance.json')
    for filename, key in [('ai_assisted_review.csv', 'public_review_sha256'),
                          ('ai_assisted_summary.json', 'summary_sha256'),
                          ('ai_assisted_review.md', 'readable_summary_sha256'),
                          ('cases.csv', 'cases_sha256'), ('sampling_design.json', 'sampling_design_sha256')]:
        path = audit / filename
        if checksum(path) != provenance[key]:
            raise ValueError(f'AI audit source hash mismatch: {filename}')
        inputs[str(path)] = checksum(path)
    if provenance['test_npz_sha256'] != checksum(OUT / 'predictions/test.npz'):
        raise ValueError('AI audit belongs to different test scores')
    if provenance['thresholds_sha256'] != checksum(OUT / 'thresholds.json'):
        raise ValueError('AI audit belongs to different thresholds')
    summary = read_json(audit / 'ai_assisted_summary.json')
    review = pd.read_csv(audit / 'ai_assisted_review.csv', dtype=str, keep_default_na=False)
    cases = pd.read_csv(audit / 'cases.csv', dtype=str, keep_default_na=False)
    if len(review) != 62 or review.review_id.tolist() != cases.review_id.tolist() or summary['ai_reviewed_rows'] != 62:
        raise ValueError('Expected the complete verified 62-case AI audit')
    if read_json(audit / 'summary.json')['completed_rows'] != 0:
        raise ValueError('Human review status changed; revise the manuscript disclosure before building')
    for filename in ('ai_assisted_provenance.json', 'summary.json'):
        inputs[str(audit / filename)] = checksum(audit / filename)
    return inputs, summary


def manuscript_values(meta):
    table = lambda name: pd.read_csv(OUT / 'tables' / f'{name}.csv')
    profile = table('threshold_profile')
    run = read_json(ROOT / 'reports/kaggle_equal20_2026-11-05/evidence/28/phobert/run.json')
    cfg = read_json(ROOT / 'configs/final_thresholds.json')
    val, test = table('val_comparison'), table('test_comparison')
    data = []
    for split, name in [('train', 'Retained train'), ('val', 'Validation'), ('test', 'Test')]:
        n = cfg['data']['splits'][split]['n']
        if split == 'train':
            positives = int(profile.train_count.sum())
            neutral = int(profile.set_index('label').loc['neutral', 'train_count'])
        else:
            positives = int(table(f'{split}_comparison').query("strategy=='fixed' and scope=='all28'").iloc[0].positive_support)
            neutral = int(table(f'{split}_neutral').query("strategy=='fixed'").iloc[0].support)
        data.append({'Split': name, 'Comments': n, 'Positive labels': positives,
                     'Mean cardinality': positives/n, 'Neutral positives': neutral})
    training = run['config']['phobert']
    training_rows = [
        ('Pretrained artifact', training['checkpoint']), ('Outputs / seed', '28 / 42'),
        ('Completed / selected epoch', f"{training['epochs']} / {run['best_epoch']}"),
        ('Checkpoint selection', 'Validation emotion27 Macro-F1 at 0.5'),
        ('Optimizer / loss', 'AdamW / unweighted BCEWithLogitsLoss'),
        ('Learning rate / weight decay', f"{training['learning_rate']:.0e} / {training['weight_decay']}"),
        ('Batch / maximum tokens', f"{training['batch_size']} / {training['max_length']}"),
        ('Gradient clipping norm', '1.0'), ('Parameter count', str(run['parameter_count'])),
        ('Original training device', run['device']),
        ('Training loop time (seconds)', f"{run['train_seconds']:.3f}"),
        ('Frozen inference', 'eval mode; sigmoid; no shuffle; batch 16'),
    ]
    gap = table('generalization_gap').query("scope=='all28'").set_index(['strategy', 'metric'])
    tiers = table('test_frequency_tiers').set_index(['tier', 'strategy'])
    cardinality = table('test_cardinality').query("scope=='all28'").copy()
    cardinality['cardinality'] = cardinality.cardinality.astype(str)
    cardinality = cardinality.set_index(['cardinality', 'strategy'])
    neutral = table('test_neutral').set_index('strategy')
    emotion = test[test.scope == 'emotion27'].set_index('strategy')
    per = table('test_per_label').pivot(index='label', columns='strategy', values='f1').loc[profile.label]
    appendix = profile[['label', 'tier', 'train_count', 'threshold']].copy()
    appendix['threshold'] = appendix.threshold.map(lambda v: f'{v:.2f}')
    for s in STRATEGIES:
        appendix[NAMES[s] + ' F1'] = per[s].to_numpy()
    appendix.columns = ['Label', 'Tier', 'Train n', 'τ', 'Fixed F1', 'Global F1', 'Per-label F1']
    audit = read_json(OUT / 'human_audit/ai_assisted_summary.json')
    author_names = [a['name'] + (f" ({a['student_id']})" if a.get('student_id') else '') for a in meta['authors']]
    byline = [*author_names, *[meta[key] for key in ('group_name', 'course_section', 'affiliation') if meta[key]],
              meta['course_title'] + ' · Final Report · ' + meta['report_date']]
    return {
        'title': meta['title'], 'author_block': ' — '.join(byline),
        'data_summary': markdown_table(pd.DataFrame(data)),
        'training_details': markdown_table(pd.DataFrame(training_rows, columns=['Setting', 'Recorded value'])),
        'primary_results': comparison_table(val, test),
        'generalization_gap': markdown_table(pd.DataFrame([
            {'Decoder': NAMES[s], 'Macro-F1 gap (pp)': f"{100*gap.loc[(s, 'macro_f1'), 'gap']:+.4f}",
             'Micro-F1 gap (pp)': f"{100*gap.loc[(s, 'micro_f1'), 'gap']:+.4f}"} for s in STRATEGIES])),
        'tier_results': markdown_table(pd.DataFrame([
            {'Tier': tier, **{NAMES[s]+' F1': tiers.loc[(tier, s), 'macro_f1'] for s in STRATEGIES},
             'Δ global (pp)': f"{100*tiers.loc[(tier, 'global'), 'delta_macro_f1_vs_fixed']:+.4f}",
             'Δ per-label (pp)': f"{100*tiers.loc[(tier, 'per_label'), 'delta_macro_f1_vs_fixed']:+.4f}"}
            for tier in ('head', 'mid', 'tail')])),
        'cardinality_results': markdown_table(pd.DataFrame([
            {'Cardinality': group, 'n': int(cardinality.loc[(group, 'fixed'), 'n']),
             **{NAMES[s]+' F1': cardinality.loc[(group, s), 'macro_f1'] for s in STRATEGIES}}
            for group in ('0', '1', '2', '3', '>=4')])),
        'neutral_results': markdown_table(pd.DataFrame([
            {'Decoder': NAMES[s], 'τ': f"{neutral.loc[s, 'threshold']:.2f}",
             **{key.upper(): int(neutral.loc[s, key]) for key in ('tp', 'fp', 'fn')},
             **{key.title(): neutral.loc[s, key] for key in ('precision', 'recall', 'f1')}} for s in STRATEGIES])),
        'emotion27_results': markdown_table(pd.DataFrame([
            {'Decoder': NAMES[s], **{label: emotion.loc[s, key] for label, key in
             [('Macro-F1', 'macro_f1'), ('Micro-F1', 'micro_f1'), ('Hamming Loss', 'hamming_loss'), ('LRAP', 'lrap')]}}
            for s in STRATEGIES])),
        'audit_effects': markdown_table(pd.DataFrame([
            {'Core sample': kind, 'Already errors at fixed': audit['core_decoder_effects'][kind].get('retained_error', 0),
             'New errors vs fixed': audit['core_decoder_effects'][kind].get('introduced_error', 0), 'Total': 20}
            for kind in ('FN', 'FP')])),
        'per_label_results': markdown_table(appendix),
    }


def inline(text):
    text = escape(text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                  lambda m: f'<link href={quoteattr(m[2])} color="#20566b">' +
                  (f'[{m[1]}]' if m[1].isdigit() else m[1]) + '</link>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return re.sub(r'`([^`]+)`', r'<font name="Courier" size="8">\1</font>', text)


def render_pdf(markdown, destination, image_base):
    register_fonts()
    normal = ParagraphStyle('PaperBody', fontName='Vigo', fontSize=9.5, leading=13.5,
                            spaceAfter=6, alignment=TA_JUSTIFY, allowWidows=0, allowOrphans=0)
    heading = ParagraphStyle('PaperHeading', parent=normal, fontName='Vigo-Bold', fontSize=12,
                             leading=16, spaceBefore=11, spaceAfter=6, keepWithNext=True)
    subheading = ParagraphStyle('PaperSubheading', parent=heading, fontSize=10.2, leading=14, spaceBefore=7)
    title = ParagraphStyle('PaperTitle', parent=heading, fontSize=18, leading=23, alignment=TA_CENTER, spaceAfter=12)
    byline = ParagraphStyle('PaperByline', parent=normal, fontSize=8.5, leading=12, alignment=TA_CENTER, spaceAfter=12)
    small = ParagraphStyle('PaperTable', parent=normal, fontSize=7.3, leading=10, alignment=0, spaceAfter=0)
    caption = ParagraphStyle('PaperCaption', parent=normal, fontSize=8.1, leading=11, alignment=0, spaceAfter=7)
    table_caption = ParagraphStyle('PaperTableCaption', parent=caption, keepWithNext=True)
    code = ParagraphStyle('PaperCode', fontName='Courier', fontSize=8, leading=11, spaceAfter=8)
    width = A4[0] - 96
    story, lines, i = [], markdown.splitlines(), 0
    after_title = False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak()); i += 1; continue
        if line.startswith('```'):
            block = []; i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                block.append(lines[i]); i += 1
            story.append(Preformatted('\n'.join(block), code)); i += 1; continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+', c) for c in cells):
                    rows.append([Paragraph(inline(c), small) for c in cells])
                i += 1
            columns = len(rows[0])
            weights = [1.65] + [1] * (columns - 1) if columns > 2 else [1, 2]
            table = Table(rows, colWidths=[width*w/sum(weights) for w in weights], repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9eef1')),
                ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#3c5360')),
                ('LINEBELOW', (0, -1), (-1, -1), .4, colors.HexColor('#8c9da5')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.extend([table, Spacer(1, 9)]); continue
        image = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
        if image:
            text, relative = image.groups()
            path = Path(image_base) / relative
            w, h = ImageReader(str(path)).getSize()
            scale = min(width/w, 420/h)
            story.append(KeepTogether([Image(str(path), w*scale, h*scale), Spacer(1, 5),
                                       Paragraph(inline(text), caption), Spacer(1, 7)]))
            i += 1; continue
        level = len(line) - len(line.lstrip('#'))
        if level:
            content = line[level:].strip()
            story.append(Paragraph(inline(content), title if level == 1 else heading if level == 2 else subheading))
            after_title = level == 1
            i += 1; continue
        paragraph = [line]; i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#', '|', '![', '```', '<!--', '- ')):
            paragraph.append(lines[i].strip()); i += 1
        text = ' '.join(paragraph)
        style = byline if after_title else table_caption if text.startswith('**Table') else normal
        after_title = False
        if text.startswith('- '):
            story.append(Paragraph(inline(text[2:]), normal, bulletText='•'))
        else:
            story.append(Paragraph(inline(text), style))

    def footer(canvas, doc):
        canvas.saveState()
        # An explicit background keeps raster previews opaque as well as print-ready.
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, *A4, fill=1, stroke=0)
        canvas.setStrokeColor(colors.HexColor('#b5c0c6')); canvas.setLineWidth(.3)
        canvas.line(48, 37, A4[0]-48, 37)
        canvas.setFont('Vigo', 7.5); canvas.setFillColor(colors.HexColor('#53646d'))
        canvas.drawString(48, 25, 'ViGoEmotions | Threshold transfer | ATI Final 2026')
        canvas.drawRightString(A4[0]-48, 25, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=48, rightMargin=48,
                            topMargin=42, bottomMargin=49, title=markdown.splitlines()[0].lstrip('# '))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def docx_inline(paragraph, text):
    """Preserve the manuscript's emphasis, code and reference hyperlinks."""
    for token in re.split(r'(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))', text):
        link = re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)', token)
        if link:
            label, url = link.groups()
            hyperlink = OxmlElement('w:hyperlink')
            hyperlink.set(qn('r:id'), paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True))
            run, properties = OxmlElement('w:r'), OxmlElement('w:rPr')
            color = OxmlElement('w:color'); color.set(qn('w:val'), '20566B')
            underline = OxmlElement('w:u'); underline.set(qn('w:val'), 'single')
            properties.extend([color, underline]); run.append(properties)
            content = OxmlElement('w:t'); content.text = f'[{label}]' if label.isdigit() else label
            run.append(content); hyperlink.append(run); paragraph._p.append(hyperlink)
        else:
            bold, code = token.startswith('**'), token.startswith('`')
            run = paragraph.add_run(token[2:-2] if bold else token[1:-1] if code else token)
            if bold:
                run.bold = True
            if code:
                run.font.name = 'Consolas'; run.font.size = Pt(8)


def render_docx(markdown, destination, image_base):
    """Export editable Word text/tables and embedded figures from the same paper."""
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Mm(210), Mm(297)
    section.left_margin = section.right_margin = Pt(48)
    section.top_margin, section.bottom_margin, section.footer_distance = Pt(42), Pt(49), Pt(20)
    body = document.styles['Normal']
    body.font.name, body.font.size = 'Arial', Pt(9.5)
    body.paragraph_format.line_spacing = Pt(13.5)
    body.paragraph_format.space_after = Pt(6)
    body.paragraph_format.widow_control = True
    for name, size in [('Title', 18), ('Heading 1', 12), ('Heading 2', 10.2)]:
        style = document.styles[name]
        style.font.name, style.font.size = 'Arial', Pt(size)
        style.font.color.rgb = RGBColor.from_string('17232D')
        style.paragraph_format.keep_with_next = True
    document.core_properties.title = markdown.splitlines()[0].lstrip('# ')
    document.core_properties.author = ''
    document.core_properties.subject = 'ViGoEmotions final report: threshold transfer'
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run('ViGoEmotions | Threshold transfer | ATI Final 2026 | ')
    run.font.size = Pt(7.5)
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    width = section.page_width - section.left_margin - section.right_margin
    lines, i, after_title = markdown.splitlines(), 0, False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line == '<!-- pagebreak -->':
            document.add_page_break(); i += 1; continue
        if line.startswith('```'):
            block = []; i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                block.append(lines[i]); i += 1
            p = document.add_paragraph()
            run = p.add_run('\n'.join(block)); run.font.name = 'Consolas'; run.font.size = Pt(8)
            i += 1; continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+', c) for c in cells):
                    rows.append(cells)
                i += 1
            table = document.add_table(rows=0, cols=len(rows[0]))
            table.style = 'Light Shading Accent 1'; table.autofit = False
            weights = [1.65] + [1] * (len(rows[0])-1) if len(rows[0]) > 2 else [1, 2]
            widths = [int(width*w/sum(weights)) for w in weights]
            for column, column_width in zip(table.columns, widths):
                column.width = column_width
            for row_index, values in enumerate(rows):
                row = table.add_row()
                row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
                if row_index == 0:
                    row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
                for cell, value, column_width in zip(row.cells, values, widths):
                    cell.width = column_width
                    p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(2)
                    p.paragraph_format.space_before = Pt(2); p.paragraph_format.line_spacing = Pt(10)
                    docx_inline(p, value)
                    for run in p.runs:
                        run.font.size = Pt(7.3)
                        if row_index == 0:
                            run.bold = True
            continue
        image = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
        if image:
            caption, relative = image.groups()
            path = Path(image_base) / relative
            w, h = ImageReader(str(path)).getSize()
            scale = min(width / 12700 / w, 420/h)
            p = document.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(path), width=Pt(w*scale), height=Pt(h*scale))
            p = document.add_paragraph(style='Caption'); docx_inline(p, caption)
            i += 1; continue
        level = len(line) - len(line.lstrip('#'))
        if level:
            p = document.add_paragraph(style='Title' if level == 1 else f'Heading {min(level-1, 2)}')
            docx_inline(p, line[level:].strip())
            if level == 1:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            after_title = level == 1
            i += 1; continue
        paragraph = [line]; i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#', '|', '![', '```', '<!--', '- ')):
            paragraph.append(lines[i].strip()); i += 1
        text = ' '.join(paragraph)
        p = document.add_paragraph(style='List Bullet' if text.startswith('- ') else 'Normal')
        docx_inline(p, text[2:] if text.startswith('- ') else text)
        if after_title:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif text.startswith('**Table'):
            p.paragraph_format.keep_with_next = True
        elif not text.startswith('- '):
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        after_title = False
    document.save(destination)


def build(destination=ROOT / 'reports'):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    inputs, audit_summary = evidence()
    template_path = ROOT / 'docs/final_paper_template.md'
    metadata_path = ROOT / 'docs/final_paper_metadata.json'
    metadata = read_json(metadata_path)
    values = manuscript_values(metadata)
    markdown = fill_template(template_path.read_text(), values)
    md, pdf = destination / 'final_report.md', destination / 'final_report.pdf'
    md.write_text(markdown, encoding='utf-8')
    render_pdf(markdown, pdf, ROOT / 'reports')
    docx = destination / 'final_report.docx'
    render_docx(markdown, docx, ROOT / 'reports')
    reader = PdfReader(pdf)
    text = '\n'.join(page.extract_text() or '' for page in reader.pages)
    sections = ('Abstract', 'Introduction', 'Data', 'Method', 'Experiments', 'Results',
                'Discussion', 'Limitations', 'Conclusion', 'References')
    if not all(section in text for section in sections) or any(token in text for token in ('{{', 'TODO', 'CHƯA CUNG CẤP')):
        raise ValueError('Incomplete manuscript/PDF export')
    for name in ('final_paper_template.md', 'final_paper_metadata.json', 'final_references.bib'):
        inputs[str(ROOT / 'docs' / name)] = checksum(ROOT / 'docs' / name)
    for path in (ROOT / 'scripts/build_report.py', ROOT / 'requirements.txt'):
        inputs[str(path)] = checksum(path)
    inputs[str(Path(__file__).resolve())] = checksum(__file__)
    for path, sha in inputs.items():
        if checksum(path) != sha:
            raise ValueError('Input changed while building the paper')
    result = dict(content_complete=True, language='en', page_count=len(reader.pages),
                  phobert_inference_performed=False, private_text_read=False,
                  audit_type='AI-assisted; no independent human adjudication',
                  ai_reviewed_comments=audit_summary['ai_reviewed_rows'], human_confirmed_comments=0,
                  rendering_packages={name: importlib.metadata.version(name) for name in ('reportlab', 'pypdf', 'python-docx', 'matplotlib', 'numpy', 'pandas')},
                  author_metadata_complete=bool(metadata['authors']) and all(a.get('name') and a.get('student_id') for a in metadata['authors']),
                  missing_identity_fields=[key for key in ('authors', 'group_name', 'course_section') if not metadata[key]],
                  input_sha256={str(Path(p).relative_to(ROOT)): sha for p, sha in inputs.items()},
                  generated_sha256={'final_report.md': checksum(md), 'final_report.pdf': checksum(pdf),
                                    'final_report.docx': checksum(docx)})
    (destination / 'final_report_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports')
    result = build(parser.parse_args().output_dir)
    print(json.dumps({key: result[key] for key in ('content_complete', 'page_count', 'audit_type', 'missing_identity_fields')}, indent=2))
