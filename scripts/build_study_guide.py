"""Render the Vietnamese study guide; never run training or rewrite reports.

Usage: .venv/bin/python scripts/build_study_guide.py
Requires reportlab, matplotlib, pypdf and pygments (PDF syntax highlighting).
"""
from pathlib import Path
import json
import os
import re
from xml.sax.saxutils import escape

os.environ.setdefault('MPLCONFIGDIR', '/tmp/vigo-study-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/vigo-study-cache')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pygments import lex
from pygments.lexers import PythonLexer, TextLexer
from pygments.styles import get_style_by_name
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, PageBreak, Paragraph,
    Spacer, Table, TableStyle, Image,
)
from reportlab.platypus.tableofcontents import TableOfContents
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
SOURCE = DOCS / 'VIGOEMOTIONS_3_MODELS_STUDY_GUIDE_VI.md'
OUTPUT = SOURCE.with_suffix('.pdf')
ASSETS = DOCS / 'study_guide_assets'
WIDTH, HEIGHT = 595.276, 841.89
MARGIN = 43
BODY_WIDTH = WIDTH - 2 * MARGIN
INK = colors.HexColor('#17354b')
TEAL = colors.HexColor('#087e8b')


def register_fonts():
    folder = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    for name, file in [('Vigo', 'DejaVuSans.ttf'), ('Vigo-Bold', 'DejaVuSans-Bold.ttf'),
                       ('VigoMono', 'DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(folder / file)))
    pdfmetrics.registerFontFamily('Vigo', normal='Vigo', bold='Vigo-Bold',
                                italic='Vigo', boldItalic='Vigo-Bold')
    pdfmetrics.registerFontFamily('VigoMono', normal='VigoMono', bold='VigoMono')


def verify_sources(markdown):
    """Every quoted repository block must still match its declared line range."""
    pattern = r'<!-- source:([^:]+):(\d+):(\d+) -->\n```(?:python|json)\n(.*?)\n```'
    checked = 0
    for path, first, last, code in re.findall(pattern, markdown, re.S):
        lines = (ROOT / path).read_text(encoding='utf-8').splitlines()
        expected = '\n'.join(lines[int(first)-1:int(last)])
        if code != expected:
            raise ValueError(f'Stale verbatim quote: {path}:{first}-{last}')
        checked += 1
    if checked < 15:
        raise ValueError('Missing annotated source quotations')
    for path, first, last in re.findall(r'`((?:scripts|configs)/[^` :]+):(\d+)(?:–(\d+))?`', markdown):
        lines = (ROOT / path).read_text(encoding='utf-8').splitlines()
        if not 1 <= int(first) <= int(last or first) <= len(lines):
            raise ValueError(f'Invalid source range: {path}:{first}-{last}')
    for char in set(markdown):
        if char.isprintable() and ord(char) not in pdfmetrics.getFont('Vigo').face.charWidths:
            raise ValueError(f'Missing Vietnamese/body glyph: {char!r}')
    return checked


def make_curve():
    ASSETS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for model, label, color in [('gru', 'BiGRU', '#087e8b'), ('phobert', 'PhoBERT 2e-5', '#d46b36')]:
        history = json.loads((ROOT / f'reports/{model}_run.json').read_text())['history']
        for ax, key in zip(axes, ('loss', 'macro_f1')):
            ax.plot([h['epoch'] for h in history], [h[key] for h in history],
                    'o-', label=label, color=color, linewidth=1.8, markersize=4)
    for ax, title in zip(axes, ('BCE trên train', 'Macro-F1 validation tại 0,5')):
        ax.set(title=title, xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / 'baseline_learning_curves.png', dpi=180)
    plt.close(fig)


def rich(value):
    value = escape(value)
    links = []
    def link(m):
        label, url = m.groups()
        links.append(f'<link href="{url}" color="#087e8b">{label}</link>')
        return f'@@LINK{len(links)-1}@@'
    value = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, value)
    value = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', value)
    value = re.sub(r'`([^`]+)`', r'<font color="#07566e">\1</font>', value)
    for i, item in enumerate(links):
        value = value.replace(f'@@LINK{i}@@', item)
    return value


def styles():
    body = ParagraphStyle('body', fontName='Vigo', fontSize=9.15, leading=13.2,
                          textColor=INK, spaceAfter=5.0, splitLongWords=True)
    return {
        'body': body,
        'title': ParagraphStyle('title', parent=body, fontName='Vigo-Bold', fontSize=29,
                                leading=38, spaceAfter=22, textColor=INK),
        'h1': ParagraphStyle('h1', parent=body, fontName='Vigo-Bold', fontSize=18,
                             leading=24, spaceAfter=12, keepWithNext=True),
        'h2': ParagraphStyle('h2', parent=body, fontName='Vigo-Bold', fontSize=12,
                             leading=17, spaceBefore=5, spaceAfter=7, keepWithNext=True),
        'h3': ParagraphStyle('h3', parent=body, fontName='Vigo-Bold', fontSize=10,
                             leading=14.2, spaceBefore=4, spaceAfter=5, keepWithNext=True),
        'small': ParagraphStyle('small', parent=body, fontSize=7.7, leading=11, spaceAfter=4),
        'cell': ParagraphStyle('cell', parent=body, fontSize=8, leading=11.1, spaceAfter=0),
        'code': ParagraphStyle('code', fontName='VigoMono', fontSize=6.65, leading=9.1,
                               textColor=INK, spaceAfter=0),
    }


class Guide(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=(WIDTH, HEIGHT), leftMargin=MARGIN,
                         rightMargin=MARGIN, topMargin=48, bottomMargin=43,
                         title='ViGoEmotions — Cẩm nang ba mô hình',
                         author='Tài liệu học tập từ mã nguồn ViGoEmotions')
        self.addPageTemplates(PageTemplate(id='guide', frames=[Frame(
            MARGIN, 43, BODY_WIDTH, HEIGHT-91, leftPadding=0, rightPadding=0,
            topPadding=0, bottomPadding=0)], onPage=self.decorate))
        self.heading_index = 0

    def beforeDocument(self):
        self.heading_index = 0

    def decorate(self, c, doc):
        c.saveState()
        c.setFillColor(colors.white)
        c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
        c.setStrokeColor(TEAL)
        c.setLineWidth(.7)
        c.line(MARGIN, HEIGHT-29, WIDTH-MARGIN, HEIGHT-29)
        c.setFont('Vigo', 7)
        c.setFillColor(INK)
        c.drawString(MARGIN, HEIGHT-22, 'ViGoEmotions  /  Cẩm nang đọc mã và bảo vệ giữa kỳ')
        c.setFont('Vigo', 7)
        c.drawString(MARGIN, 23, 'Đối chiếu mã nguồn: 23/10/2026  •  Kết quả validation')
        c.drawRightString(WIDTH-MARGIN, 23, f'{doc.page:02d}')
        c.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and getattr(flowable, 'toc_heading', False):
            self.heading_index += 1
            key = f'chapter-{self.heading_index}'
            label = flowable.getPlainText()
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(label, key, level=0)
            self.notify('TOCEntry', (0, label, self.page, key))


def code_table(code, language, style):
    palette = get_style_by_name('friendly')
    lines = ['']
    lexer = PythonLexer() if language == 'python' else TextLexer()
    for token, text in lex(code, lexer):
        color = palette.style_for_token(token).get('color') or '17354b'
        for j, part in enumerate(text.split('\n')):
            if j:
                lines.append('')
            if part:
                lines[-1] += f'<font color="#{color}">{escape(part).replace(" ", "&#160;")}</font>'
    if lines and not lines[-1]:
        lines.pop()
    # Paragraph wraps long statements, preserving all text and syntax colors.
    rows = [[Paragraph(line or '&#160;', style['code'])] for line in lines]
    table = Table(rows, colWidths=[BODY_WIDTH-2], hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f2f5f8')),
        ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, 0), 5), ('BOTTOMPADDING', (0, -1), (-1, -1), 5),
    ]))
    return [table, Spacer(1, 5)]


def parse(markdown, style):
    story, lines, i = [], markdown.splitlines(), 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == '<!-- page -->':
            story.append(PageBreak())
        elif line == '<!-- toc -->':
            toc = TableOfContents()
            toc.levelStyles = [ParagraphStyle('toc', parent=style['body'], leading=20,
                                              fontSize=9.2, spaceBefore=2)]
            toc.dotsMinLevel = 0
            story.extend([toc, Spacer(1, 15)])
        elif line.startswith('<!--'):
            continue
        elif line.startswith('```'):
            language = line[3:]
            block = []
            while i < len(lines) and not lines[i].startswith('```'):
                block.append(lines[i]); i += 1
            i += 1
            if language == 'flow':
                cells = []
                for item in block:
                    cells.append([Paragraph(rich(item), style['body'])])
                t = Table(cells, colWidths=[BODY_WIDTH], hAlign='LEFT')
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e9f4f5')),
                    ('BOX', (0, 0), (-1, -1), .5, TEAL),
                    ('LINEBELOW', (0, 0), (-1, -2), .3, colors.HexColor('#b2d5da')),
                    ('LEFTPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.extend([t, Spacer(1, 8)])
            else:
                story.extend(code_table('\n'.join(block), language, style))
        elif line.startswith('|'):
            rows = [line]
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip()); i += 1
            data = []
            for row in rows:
                values = [v.strip() for v in row.strip('|').split('|')]
                if all(re.fullmatch(r'[:\- ]+', v) for v in values):
                    continue
                data.append(values)
            n = len(data[0])
            weights = [1.2]+[1]*(n-1) if n > 2 else [1, 2.5]
            widths = [BODY_WIDTH * w/sum(weights) for w in weights]
            t = Table([[Paragraph(rich(c), style['cell']) for c in row] for row in data],
                      colWidths=widths, repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dceef1')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f4f7fa')]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LINEBELOW', (0, 0), (-1, 0), .6, TEAL),
                ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.extend([t, Spacer(1, 8)])
        elif line.startswith('!['):
            caption, path = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line).groups()
            picture = Image(str(DOCS / path))
            scale = min(BODY_WIDTH/picture.imageWidth, 205/picture.imageHeight)
            picture.drawWidth = picture.imageWidth*scale
            picture.drawHeight = picture.imageHeight*scale
            story.extend([picture, Paragraph(rich(caption), style['small']), Spacer(1, 7)])
        elif line.startswith('#'):
            hashes = len(line)-len(line.lstrip('#'))
            text = line[hashes:].strip()
            p = Paragraph(rich(text), style[{1:'title', 2:'h1', 3:'h2', 4:'h3'}[hashes]])
            p.toc_heading = hashes == 2 and text != 'Mục lục và cách sử dụng'
            story.append(p)
        elif line.startswith('> '):
            qstyle = ParagraphStyle('quote_cell', parent=style['body'], spaceAfter=0)
            quote = Table([[Paragraph(rich(line[2:]), qstyle)]], colWidths=[BODY_WIDTH], hAlign='LEFT')
            quote.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eff7f8')),
                ('BOX', (0, 0), (-1, -1), .6, TEAL),
                ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ]))
            story.extend([Spacer(1, 3), quote, Spacer(1, 8)])
        else:
            text = line
            while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||```|<!--|> |!\[|- |\d+\. )', lines[i].strip()):
                text += ' ' + lines[i].strip(); i += 1
            if text.startswith('- '):
                text = '• ' + text[2:]
            story.append(Paragraph(rich(text), style['small'] if text.startswith('Nguồn [MÃ]:') else style['body']))
    return story


def main():
    register_fonts()
    markdown = SOURCE.read_text(encoding='utf-8')
    quotes = verify_sources(markdown)
    make_curve()
    Guide(OUTPUT).multiBuild(parse(markdown, styles()))
    reader = PdfReader(OUTPUT)
    text = '\n'.join(page.extract_text() or '' for page in reader.pages)
    assert len(text) > 40000 and 'tiếng Việt' in text and 'Macro-F1' in text
    assert '\ufffd' not in text and len(reader.outline) >= 9
    print(f'{OUTPUT}\nPages: {len(reader.pages)}; verified code blocks: {quotes}; extractable characters: {len(text)}')


if __name__ == '__main__':
    main()
