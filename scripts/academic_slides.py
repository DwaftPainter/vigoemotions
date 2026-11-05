"""Vector slide primitives and reusable academic layouts (16:9, no raster charts)."""
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

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

    def begin(self, slide, index):
        self.current = index
        self.rect(0, 0, W, H)
        self.label(slide['section'], 64, 686)
        self.text(slide['title'], 64, 654, 1152, 32, NAVY, True, height=48)
        self.text(slide['subtitle'], 64, 603, 1152, 16, MUTED, height=42)
        self.line(64, 548, 1216, 548, LINE)
        self.line(64, 548, 124, 548, TEAL, 3)

    def takeaway(self, value):
        self.rect(64, 68, 1152, 62, TINT, radius=4)
        self.label('Takeaway', 82, 111, 110)
        self.text(value, 208, 112, 982, 18, NAVY, True, height=47)

    def finish(self, slide, index):
        self.line(64, 48, 1216, 48)
        self.text(slide['source'], 64, 34, 1060, 10, MUTED, height=24)
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

    def card(self, x, top, width, height, title, body, accent=TEAL):
        self.rect(x, top-height, width, height, PALE, radius=5)
        self.rect(x, top-height, 4, height, accent)
        self.label(title, x+20, top-18, width-40, accent)
        if body:
            self.text(body, x+20, top-50, width-40, 19, INK, height=height-64)

    def stat(self, x, top, width, number, label, color=TEAL):
        self.text(number, x, top, width, 36, color, True, height=50)
        self.text(label, x, top-54, width, 15, MUTED, height=42)

    def arrow(self, x1, y, x2, color=TEAL):
        self.line(x1, y, x2, y, color, 1.5)
        self.line(x2-6, y+4, x2, y, color, 1.5)
        self.line(x2-6, y-4, x2, y, color, 1.5)

    def node(self, x, top, w, h, heading, body, fill=PALE):
        self.rect(x, top-h, w, h, fill, LINE, radius=4)
        self.text(heading, x+12, top-14, w-24, 17, NAVY, True, height=42, align=1)
        if body:
            self.text(body, x+12, top-51, w-24, 14, MUTED, height=h-60, align=1)

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
            formatted = (f'{value:.4f}' if value_format == 'f1' else
                         f'{value:.2%}' if value_format == 'percent' else f'{value:,.0f}')
            self.text(formatted, bar_x+bar_w+9, y-7, 62, 14, INK, True,
                      height=25, align=2)

    def plot(self, series, x, y, width, height, xlim, ylim, xticks, yticks,
             xlabel, ylabel, markers=(), scatter=False, show_legend=True):
        left, right, bottom, top = x+56, x+width-18, y+50, y+height-60
        px = lambda value:left+(value-xlim[0])/(xlim[1]-xlim[0])*(right-left)
        py = lambda value:bottom+(value-ylim[0])/(ylim[1]-ylim[0])*(top-bottom)
        self.label(ylabel, x+4, y+height-5, width-8, MUTED)
        for tick in yticks:
            self.line(left, py(tick), right, py(tick))
            self.text(f'{tick:g}', x, py(tick)+7, 42, 12, MUTED, align=2, height=20)
        for tick in xticks:
            self.text(f'{tick:g}', px(tick)-28, bottom-11, 56, 12, MUTED, align=1, height=20)
        self.line(left, bottom, right, bottom, MUTED)
        self.text(xlabel, left, y+10, right-left, 13, MUTED, align=1, height=20)
        for s in series:
            points = [(px(a),py(b)) for a,b in zip(s['x'],s['y'])]
            for a,b in points:
                if not (left-.1 <= a <= right+.1 and bottom-.1 <= b <= top+.1):
                    raise ValueError('Plot data outside declared scale')
            if not scatter:
                for a,b in zip(points,points[1:]):self.line(*a,*b,s['color'],2.3,dash=s.get('dash'))
            for a,b in points:
                self.c.setFillColor(colors.HexColor(s['color']))
                self.c.circle(a,b,3 if scatter else 2.6,fill=1,stroke=0)
        for marker in markers:
            sx,sy=px(marker['x']),py(marker['y'])
            self.line(sx,bottom,sx,sy,marker['color'],1,dash=(3,3))
            self.c.setFillColor(colors.white);self.c.setStrokeColor(colors.HexColor(marker['color']))
            self.c.circle(sx,sy,6,fill=1,stroke=1)
            if marker.get('label'):
                self.text(marker['label'],sx-72,sy+31,144,12,marker['color'],True,height=20,align=1)
        if show_legend:
            cursor=left
            for s in series:
                self.line(cursor,top+18,cursor+22,top+18,s['color'],2.5,dash=s.get('dash'))
                self.text(s['name'],cursor+30,top+26,155,12,MUTED,height=20)
                cursor += 190


def render_title(p, s):
    p.label('ATI 2026 / MACHINE LEARNING MIDTERM',64,678,900)
    p.text('ViGoEmotions',64,614,1120,58,NAVY,True,height=85)
    p.text('Fine-grained emotion recognition\nin Vietnamese social media',67,519,1000,32,INK,height=95)
    p.line(64,393,1216,393)
    p.text(s['headline'],64,358,760,25,TEAL,True,height=80)
    for i,stat in enumerate(s['stats']):p.stat(64+i*385,226,320,*stat)
    p.text(s['subtitle'],877,352,339,17,MUTED,height=115)


def render_table(p,s):
    t=s['table'];bottom=p.table(**t)
    if s.get('caption'):p.text(s['caption'],64,bottom-18,1152,14,MUTED,height=58)
    for card in s.get('cards',[]):p.card(**card)


def render_table_chart(p,s):
    p.table(**s['table'])
    if 'bars' in s:p.bar_chart(**s['bars'])
    if 'scatter' in s:p.plot(**s['scatter'])
    for stat in s.get('stats',[]):p.stat(**stat)
    for item in s.get('annotations',[]):p.text(**item)


def render_encoding(p,s):
    p.table(**s['table'])
    p.label('Same input, different target space',64,232,800)
    p.node(64,199,254,50,'Target: joy + Neutral','',PALE)
    p.arrow(334,174,381)
    p.text('27 outputs: joy = 1',400,193,340,18,NAVY,True,height=32)
    p.text('28 outputs: joy = 1, neutral = 1',782,193,434,18,TEAL,True,height=32)


def render_pipeline(p,s):
    for row in s['routes']:
        p.label(row['name'],64,row['top']-17,170,row['color'])
        p.text(row['tag'],64,row['top']-49,165,14,MUTED,height=50)
        for i,node in enumerate(row['nodes']):
            x=250+i*246;p.node(x,row['top'],220,100,node[0],node[1],TINT if i==3 else PALE)
            if i<3:p.arrow(x+224,row['top']-50,x+242,row['color'])
    p.text(s['caption'],64,171,1152,15,MUTED,height=33)


def render_paired(p,s):
    p.table(**s['table'])
    p.label('Shared-emotion Macro-F1',794,520,422)
    for j,(name,color) in enumerate([('27 outputs',NAVY),('28 → 27',TEAL)]):
        x=794+j*210
        p.line(x,478,x+22,478,color,2.5)
        p.text(name,x+31,486,160,12,MUTED,height=20)
    left,right=946,1200
    px=lambda value:left+(value-.2)/.45*(right-left)
    for i,row in enumerate(s['comparison']):
        y=425-i*73
        p.text(row['model'],794,y+10,140,14,INK,True,height=24)
        values=row['values'];a,b=px(values[0][1]),px(values[1][1])
        p.line(a,y,b,y,BLUE,3)
        p.c.setStrokeColor(colors.HexColor(NAVY));p.c.setFillColor(colors.white)
        p.c.circle(a,y,6,fill=1,stroke=1)
        p.c.setFillColor(colors.HexColor(TEAL));p.c.circle(b,y,3.5,fill=1,stroke=0)
    p.line(left,250,right,250,MUTED)
    for tick in (.2,.4,.6):p.text(f'{tick:.1f}',px(tick)-20,239,40,12,MUTED,align=1,height=20)
    p.text(s['caption'],64,200,1152,16,MUTED,height=60)


def render_neutral(p,s):
    p.table(**s['table'])
    for stat in s['stats']:p.stat(**stat)
    p.text(s['caption'],64,158,1152,14,MUTED,height=25)


def render_curves(p,s):
    for chart in s['charts']:
        p.text(chart['title'],chart['x'],523,chart['width'],21,NAVY,True,height=32)
        args={k:v for k,v in chart.items() if k!='title'};p.plot(**args)
    p.text(s['caption'],64,179,1152,15,MUTED,height=35)


def render_threshold(p,s):
    p.plot(**s['chart'])
    for stat in s['stats']:p.stat(**stat)
    p.text(s['caption'],64,173,1152,15,MUTED,height=35)


def render_comparison(p,s):
    for panel in s['panels']:
        x=panel['x'];p.rect(x,224,552,302,PALE,radius=5)
        p.label(panel['heading'],x+22,504,508,panel['color'])
        p.text(panel['id'],x+22,466,508,24,NAVY,True,height=40)
        p.text(panel['truth'],x+22,412,508,19,INK,height=56)
        p.text(panel['prediction'],x+22,359,508,19,INK,height=56)
        p.text(panel['probability'],x+22,307,508,17,panel['color'],True,height=30)
        p.text(panel['hypothesis'],x+22,267,508,15,MUTED,height=36)
    p.text(s['caption'],64,185,1152,15,MUTED,height=38)


def render_final_plan(p,s):
    p.rect(64,451,1152,69,TINT,radius=4)
    p.text(s['checkpoint'],84,501,1112,23,NAVY,True,height=40)
    for i,card in enumerate(s['strategies']):p.card(64+i*391,418,370,141,card[0],card[1])
    for i,node in enumerate(s['nodes']):
        x=64+i*391;p.node(x,247,370,83,node[0],node[1])
        if i<2:p.arrow(x+374,205,x+387)


def render_conclusion(p,s):
    for i,item in enumerate(s['findings']):
        top=519-i*113
        p.text(f'{i+1:02d}',64,top,70,30,TEAL,True,height=46)
        p.text(item[0],151,top,700,24,NAVY,True,height=36)
        p.text(item[1],151,top-45,700,17,MUTED,height=49)
    p.rect(908,194,308,332,PALE,radius=4)
    p.label('Scope and limitations',930,503,263)
    for i,item in enumerate(s['limits']):p.text(item,930,462-i*57,263,16,INK,height=48)


LAYOUTS={'title':render_title,'full_table':render_table,'table_chart':render_table_chart,
         'encoding':render_encoding,'pipeline':render_pipeline,'paired':render_paired,
         'neutral':render_neutral,'curves':render_curves,
         'threshold':render_threshold,'two_column':render_comparison,
         'final_plan':render_final_plan,'conclusions':render_conclusion}


def render_deck(slides, target):
    p=SlideCanvas(target,len(slides))
    for i,s in enumerate(slides,1):
        if s['layout']=='title':
            p.current=i;p.rect(0,0,W,H)
        else:p.begin(s,i)
        LAYOUTS[s['layout']](p,s)
        if s.get('takeaway'):p.takeaway(s['takeaway'])
        p.finish(s,i)
    p.c.save()
    return p.text_boxes
