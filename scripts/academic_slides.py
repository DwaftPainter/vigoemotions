"""Academic drawing primitives; render approved slide text without rewriting it."""
from xml.sax.saxutils import escape
import re

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

from common import ROOT

W, H = 1280, 720
NAVY = '#142D46'
TEAL = '#007F86'
INK = '#20384D'
MUTED = '#607487'
LINE = '#DDE6EC'
PALE = '#F2F6F8'
TINT = '#E7F4F3'
AMBER = '#A56C22'
WHITE = '#FFFFFF'
BLUE = '#597594'


class SlideCanvas:
    def __init__(self, target, count):
        from build_report import register_fonts
        register_fonts()
        self.c = Canvas(str(target), pagesize=(W, H), pageCompression=1)
        self.c.setTitle('ViGoEmotions | Midterm research presentation')
        self.c.setAuthor('ViGoEmotions project')
        self.count = count
        self.text_boxes = []
        self.current = 0

    def rect(self, x, y, w, h, fill=WHITE, stroke=None, radius=0):
        self.c.setFillColor(colors.HexColor(fill))
        self.c.setStrokeColor(colors.HexColor(stroke or fill))
        if radius:
            self.c.roundRect(x, y, w, h, radius, fill=1, stroke=int(stroke is not None))
        else:
            self.c.rect(x, y, w, h, fill=1, stroke=int(stroke is not None))

    def line(self, x1, y1, x2, y2, color=LINE, width=1, dash=None):
        self.c.saveState()
        self.c.setStrokeColor(colors.HexColor(color)); self.c.setLineWidth(width)
        if dash:
            self.c.setDash(*dash)
        self.c.line(x1, y1, x2, y2)
        self.c.restoreState()

    def text(self, value, x, top, width, size=20, color=INK, bold=False,
             height=100, align=0, markup=False):
        if not (0 <= x and x+width <= W+.01 and 0 <= top <= H):
            raise ValueError(f'Slide {self.current}: text outside page: {value}')
        content = str(value) if markup else escape(str(value)).replace('\n', '<br/>')
        style = ParagraphStyle('slide', fontName='Vigo-Bold' if bold else 'Vigo',
            fontSize=size, leading=size*1.28, textColor=colors.HexColor(color), alignment=align)
        p = Paragraph(content, style)
        _, used = p.wrap(width, H)
        if used > height+.1 or top-used < 0:
            raise ValueError(f'Slide {self.current}: text overflow ({used:.1f}>{height}): {value}')
        p.drawOn(self.c, x, top-used)
        self.text_boxes.append(dict(slide=self.current, text=str(value), x=x, y=top-used,
                                    width=width, height=used, font_size=size))
        return used

    def label(self, value, x, top, width=500, color=TEAL):
        return self.text(value.upper(), x, top, width, 12, color, True, height=20)



    def finish(self, slide, index):
        self.line(64, 48, 1216, 48)
        self.text(slide.get('source', 'ATI 2026 / ViGoEmotions / Validation only'), 64, 34, 1060, 10, MUTED, height=24)
        self.text(f'{index:02d} / {self.count:02d}', 1136, 34, 80, 11, MUTED, align=2, height=20)
        self.c.showPage()

    def table(self, headers, rows, x, top, widths, row_height=54, header_height=44,
              highlights=(), group_starts=(), font_size=16, alignments=None):
        total = sum(widths)
        alignments = alignments or [0,0]+[2]*(len(widths)-2)
        self.rect(x, top-header_height, total, header_height, NAVY)
        cursor = x
        for j, header in enumerate(headers):
            self.text(header, cursor+12, top-10, widths[j]-24, 13, WHITE, True,
                      height=header_height-14, align=alignments[j])
            cursor += widths[j]
        y = top-header_height
        for i, row in enumerate(rows):
            if len(row) != len(widths):
                raise ValueError('Table column mismatch')
            self.rect(x, y-row_height, total, row_height, WHITE if i % 2 else PALE)
            if i in group_starts:
                self.line(x, y, x+total, y, TEAL, 2)
            cursor = x
            for j, cell in enumerate(row):
                hit = (i, j) in highlights
                if hit:
                    self.rect(cursor+3, y-row_height+4, widths[j]-6, row_height-8, TINT, radius=3)
                self.text(cell, cursor+12, y-12, widths[j]-24, font_size,
                          TEAL if hit else INK, bold=hit or j == 0,
                          height=row_height-18, align=alignments[j])
                cursor += widths[j]
            y -= row_height
        self.line(x, y, x+total, y)
        return y



    def arrow(self, x1, y, x2, color=TEAL):
        self.line(x1, y, x2, y, color, 1.5)
        self.line(x2-6, y+4, x2, y, color, 1.5)
        self.line(x2-6, y-4, x2, y, color, 1.5)


    def bar_chart(self, labels, values, x, top, width, height, maximum=None,
                  value_format='count', colors_by_row=None, label_width=155):
        maxval = maximum or max(values)*1.16
        bar_x = x+label_width
        bar_w = width-label_width-72
        step = height/len(values)
        for i,(name,value) in enumerate(zip(labels,values)):
            y = top-i*step
            self.text(name, x, y-7, label_width-14, 15, INK, height=step-6)
            self.rect(bar_x, y-31, bar_w, 22, PALE)
            self.rect(bar_x, y-31, max(.3,bar_w*value/maxval), 22,
                      colors_by_row[i] if colors_by_row else TEAL)
            self.text(f'{value:.4f}' if value_format == 'f1' else f'{value:,.0f}', bar_x+bar_w+9, y-7, 62, 14, INK, True,
                      height=25, align=2)



def band(p, text, top=155, fill=TINT, size=20):
    p.rect(64, top-76, 1152, 76, fill, radius=4)
    p.text(text, 86, top-17, 1108, size, NAVY, True, height=55)


def panel(p, text, x, top, width, height, size=23, fill=PALE, color=INK, bold=False):
    p.rect(x, top-height, width, height, fill, radius=5)
    p.text(text, x+22, top-20, width-44, size, color, bold, height=height-36)


def frame(p, s, index):
    p.current=index
    p.rect(0,0,W,H)
    p.label('RESEARCH QUESTION TABLES' if s.get('research_tables') else 'ViGoEmotions / MIDTERM',64,686,1000)
    p.text(s['title'],64,650,1152,31,NAVY,True,height=80)
    p.line(64,548,1216,548)
    p.line(64,548,124,548,TEAL,3)
    if s.get('subtitle'):p.text(s['subtitle'],64,593,1152,16,MUTED,height=38)


def render_title(p,s):
    b=s['bullets']
    p.text(b[0],64,508,1110,27,NAVY,height=80)
    p.text(b[1],64,402,1110,34,TEAL,True,height=88)
    p.text(b[2],64,292,1110,22,INK,height=62)
    panel(p,b[3],64,201,552,109,22)
    panel(p,b[4],640,201,576,109,22,TINT,NAVY,True)


def render_figure(p,s):
    b=s['bullets']
    if s['approved_slide']==2:
        x,width=64,430;fx,fy,fw,fh=541,79,675,450
        top=516;step=101
    else:
        x,width=64,335;fx,fy,fw,fh=430,83,786,443
        top=516;step=84
    for i,text in enumerate(b):
        p.rect(x,top-i*step-4,4,4,TEAL)
        p.text(text,x+18,top-i*step,width-18,20,INK,height=step-12)
    path=ROOT/'reports'/s['figure'];iw,ih=ImageReader(str(path)).getSize()
    scale=min(fw/iw,fh/ih)
    p.c.drawImage(str(path),fx+(fw-iw*scale)/2,fy+(fh-ih*scale)/2,iw*scale,ih*scale,mask='auto')


def render_encoding(p,s):
    b=s['bullets']
    panel(p,b[0],64,520,552,151,25)
    panel(p,b[1],640,520,576,151,25,TINT,NAVY,True)
    panel(p,b[2],64,344,1152,69,22)
    panel(p,b[3],64,250,552,144,24)
    panel(p,b[4],640,250,576,144,24)


def render_models(p,s):
    b=s['bullets']
    for i,text in enumerate(b[:3]):
        x=64+i*392
        panel(p,text,x,520,368,231,23,TINT if i==2 else PALE,NAVY,i==2)
    panel(p,b[3],64,261,1152,78,23)
    band(p,b[4])


def macro_values(bullets):
    return [float(re.search(r'Macro-F1: (\d+\.\d+)',v).group(1)) for v in bullets[:3]]


def render_performance(p,s):
    b=s['bullets']
    for i,text in enumerate(b[:3]):panel(p,text,64,520-i*100,510,94,21,TINT if i==2 else PALE,NAVY,i==2)
    p.label('Macro-F1',620,520,596)
    p.bar_chart(['TF-IDF','BiGRU','PhoBERT'],macro_values(b),620,463,596,224,maximum=1,value_format='f1',
                label_width=135,colors_by_row=[BLUE,BLUE,TEAL])
    p.text(b[3],64,194,1152,22,NAVY,True,height=38)
    band(p,b[4],top=139,size=19)


def render_neutral_effect(p,s):
    b=s['bullets']
    panel(p,b[0],64,520,552,133,26,TINT,NAVY,True)
    panel(p,b[1],640,520,576,133,26)
    p.text(b[2],64,349,1152,32,TEAL,True,height=62)
    panel(p,b[3],64,249,1152,74,23,TINT,NAVY,True)
    band(p,b[4],fill=PALE)


def render_neutral_detection(p,s):
    b=s['bullets']
    panel(p,b[0],64,520,1152,104,30,TINT,NAVY,True)
    panel(p,b[1],64,389,552,136,25)
    panel(p,b[2],640,389,576,136,25)
    p.text(b[3],64,223,1152,23,INK,height=50)
    band(p,b[4])


def render_questions(p,s):
    b=s['bullets'];labels=['Q1','Q2'] if s['approved_slide']==10 else ['Q3','Q4']
    for i,name in enumerate(labels):
        x=64+i*592
        p.rect(x,196,560,324,PALE,radius=5)
        p.label(name,x+24,496,500)
        p.text(b[2*i],x+24,445,512,24,NAVY,True,height=100)
        p.text(b[2*i+1],x+24,320,512,24,TEAL,True,height=100)
    band(p,b[4],fill=TINT)


def render_threshold(p,s):
    b=s['bullets']
    panel(p,b[0],64,520,552,146,27)
    panel(p,b[1],640,520,576,146,27,TINT,NAVY,True)
    p.text(b[2],64,336,1152,23,INK,height=58)
    p.text(b[3],64,260,1152,25,NAVY,True,height=64)
    band(p,b[4])


def render_errors(p,s):
    b=s['bullets']
    p.text(b[0],64,520,1152,27,NAVY,True,height=70)
    for i in range(6):p.rect(68+i*60,405,38,9,TEAL if i<3 else AMBER,radius=3)
    panel(p,b[1],64,373,1152,106,25)
    panel(p,b[2],64,237,552,136,23)
    panel(p,b[3],640,237,576,136,23,TINT,NAVY,True)


def render_plan(p,s):
    b=s['bullets']
    p.text(b[0],64,520,1152,30,TEAL,True,height=70)
    for i,text in enumerate(b[1:4]):
        x=64+i*392
        panel(p,text,x,407,368,201,23,TINT if i==2 else PALE)
        if i<2:p.arrow(x+372,306,x+385)
    if len(b)>4:band(p,b[4])


def render_conclusion(p,s):
    b=s['bullets']
    p.text(b[0],64,520,1152,31,NAVY,True,height=75)
    p.text(b[1],64,419,1152,29,TEAL,True,height=75)
    panel(p,b[2],64,307,552,131,22)
    panel(p,b[3],640,307,576,131,22)
    p.text(b[4],64,140,1152,18,MUTED,height=58)


def render_question_tables(p,s):
    for table in s['tables']:
        p.label(table['question_title'],table['x'],519,sum(table['widths']))
        p.table(**{key:value for key,value in table.items() if key not in ('rq','question_title')})
    p.text(s['caption'],64,174,1152,16,MUTED,height=93)


LAYOUTS={'title':render_title,'figure':render_figure,'encoding':render_encoding,'models':render_models,
         'performance':render_performance,'neutral_effect':render_neutral_effect,
         'neutral_detection':render_neutral_detection,'questions':render_questions,
         'threshold':render_threshold,'errors':render_errors,'plan':render_plan,'conclusion':render_conclusion,
         'question_tables':render_question_tables}


def render_deck(slides,target):
    p=SlideCanvas(target,len(slides))
    for i,s in enumerate(slides,1):
        frame(p,s,i)
        LAYOUTS[s['layout']](p,s)
        p.finish(s,i)
    p.c.save()
    return p.text_boxes
