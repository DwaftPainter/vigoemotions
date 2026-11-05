"""English midterm slides and a matching rehearsal script; no model execution."""
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, KeepTogether

from common import ROOT, write_json

SCRIPT = ROOT/'docs/midterm_presentation_en.md'


def presentation_content():
    from academic_slide_design import slide_definitions
    return slide_definitions()


def questions():
    return [
        ('Why use sigmoid rather than softmax?', 'Several emotions can be positive together. Independent sigmoid outputs do not force the probabilities to sum to one.'),
        ('Why use binary cross-entropy?', 'Each output is a binary emotion decision. BCEWithLogitsLoss combines the sigmoid calculation with binary cross-entropy in a numerically stable training loss.'),
        ('Did both neural models really train for 20 epochs?', 'Yes. All four neural histories contain epochs 1 through 20. The selected checkpoint can come from an earlier epoch.'),
        ('Does adding Neutral help?', 'Not on the shared 27-emotion F1 in this run. PhoBERT scores 0.5974 with 27 outputs, versus 0.5836 when the 28-output predictions are scored on those same emotions. One seed is insufficient for a general causal claim.'),
        ('Why are the 28-label scores 0.5783 and 0.5836 different?', 'They use the same checkpoint. The first evaluates all 28 labels; the second evaluates only the 27 emotion columns.'),
        ('Did you prove that the supplied split was stratified?', 'No. We verified integrity and described label-distribution differences, but did not verify the original splitting algorithm. Acceptance of the supplied split still needs clarification.'),
        ('Are these test results?', 'No. The model scores are from validation. Test data participated in the data-integrity audit, but was not used to evaluate models or tune thresholds.'),
        ('Why did F1 start at zero?', 'At the first epoch, there were no true-positive label predictions at threshold 0.5. Later epochs improved. That single early result did not establish a failed run.'),
        ('Did you use both T4 GPUs?', 'The code uses one GPU per training process, even though Kaggle lists T4 times two. No distributed training was configured.'),
        ('Where do the pretrained weights come from?', 'PhoBERT-base-v2 is loaded from the VinAI checkpoint. Its backbone is pretrained, while the classification head is initialized for our labels. Fine-tuning updates both. BiGRU and Logistic Regression are trained from scratch.'),
        ('Why do some file hashes differ between runs?', 'We verified that the differences are exact LF versus CRLF line-ending variants of the same local files. Parsed configurations and fixed split IDs match; original manifests were preserved.'),
        ('Is the threshold 0.35 now your final choice?', 'No. It is the best point in an exploratory validation grid for a fixed checkpoint. The main comparison remains at 0.5, and final decoding choices have not been evaluated on test.'),
    ]


def script_markdown(slides):
    words = sum(len(re.findall(r"\b[\w'-]+\b", speech)) for speech in (slide['notes'] for slide in slides))
    lines = ['# ViGoEmotions — English Midterm Presentation Script', '',
        f'{len(slides)} slides; {words:,} spoken words. Target: about 8–9 minutes, including short pauses and handovers. Finish before the 10-minute limit; leave any remaining time for questions.', '',
        'Use this script with reports/midterm_slides.pdf. The numerical results come from the verified Kaggle 20-epoch studies. The Vietnamese report and study guide remain available for preparation.', '',
        '## Suggested timing', '',
        '- Slides 1–5: introduction, data and model setup, about 2 minutes.',
        '- Slides 6–9: results, Neutral comparison and learning curves, about 2 minutes.',
        '- Slides 10–15: dedicated Q1–Q5 analyses, about 3–3.5 minutes.',
        '- Slides 16–18: error analysis, final research question and conclusion, about 1.5 minutes.', '',
        '## Before presenting', '',
        '- Add your real group details to the submitted materials; do not read these preparation instructions aloud.',
        '- Confirm the six AI-assisted error-review observations before presenting them as your team’s analysis.',
        '- Rehearse once with a timer. Read key results, rather than every number on a slide.',
        '- Possible handover points: after slide 5 (setup) and slide 9 (results). Allocate these sections to your actual team members.',
        '- Rehearse at roughly 110–120 words per minute. If behind schedule, shorten the learning-curve and frequency comments; keep the main results and conclusion.', '',
        '## Slide-by-slide script', '']
    for i, slide in enumerate(slides, 1):
        title, speech = slide['title'], slide['notes']
        lines.extend([f'### Slide {i:02d} — {title}', '', speech, ''])
        if i in (5, 9):
            transition = {5: 'That is our setup. My teammate will now present the results.',
                          9: 'My teammate will now discuss the analysis and next steps.'}[i]
            lines.extend([f'Optional handover: “{transition}”', ''])
    lines.extend(['## Short answers for Q&A', ''])
    for question, answer in questions():
        lines.extend([f'### {question}', '', answer, ''])
    lines.extend(['## Speaking notes', '',
        '- TF-IDF: say the letters “T F, I D F”. OvR means “one versus rest”.',
        '- BiGRU: “bidirectional G R U”. BCE: “binary cross-entropy”. LRAP: “label ranking average precision”.',
        '- Read 0.5974 as “zero point five nine seven four”. Keep decimal points in English slides.',
        '- A change of 0.0138 in an F1 score is about 1.38 percentage points; do not call it a 1.38 percent relative change.',
        '- Say “validation results”, “in this run”, and “preliminary review” where appropriate. Do not claim test performance or statistical significance.', ''])
    return '\n'.join(lines), words


def render_script(markdown):
    from build_report import register_fonts
    register_fonts()
    normal = ParagraphStyle('EnglishScript', fontName='Vigo', fontSize=10.5, leading=15,
                            spaceAfter=8, textColor=colors.HexColor('#17354b'))
    heading = ParagraphStyle('EnglishScriptHeading', parent=normal, fontName='Vigo-Bold',
                             fontSize=13, leading=18, spaceBefore=9, keepWithNext=True)
    title = ParagraphStyle('EnglishScriptTitle', parent=heading, fontSize=20, leading=27)
    story, notes_start = [], None
    for line in markdown.splitlines():
        if not line.strip():
            continue
        level = len(line)-len(line.lstrip('#'))
        text = line[level:].strip() if level else line
        style = title if level == 1 else heading if level else normal
        if line == '## Speaking notes':
            notes_start = len(story)
        story.append(Paragraph(escape(text), style))
    if notes_start is not None:
        story = story[:notes_start]+[KeepTogether(story[notes_start:])]
    def footer(c, doc):
        c.saveState()
        c.setFillColor(colors.white)
        c.rect(0, 0, 595, 842, fill=1, stroke=0)
        c.setFont('Vigo', 8); c.setFillColor(colors.HexColor('#607684'))
        c.drawString(45, 24, 'ViGoEmotions | English rehearsal script | Validation results')
        c.drawRightString(550, 24, str(doc.page))
        c.restoreState()
    doc = SimpleDocTemplate(str(SCRIPT.with_suffix('.pdf')), pagesize=(595, 842),
                            leftMargin=45, rightMargin=45, topMargin=38, bottomMargin=44,
                            title='ViGoEmotions — English Midterm Script')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def slide_markdown(slides):
    lines=['# ViGoEmotions — Midterm research presentation', '',
           f'{len(slides)} slides • English • validation results from the verified Kaggle 20-epoch studies.', '',
           'Experiment inputs are read-only. Slide definitions: reports/midterm_slide_design.json.', '']
    for i,s in enumerate(slides,1):
        lines += [f"## {i}. {s['title']}", '', f"**Layout:** {s['layout']} · **Section:** {s['section']}", '',
                  s['subtitle'], '']
        if s.get('headline'):lines += [s['headline'].replace('\n',' '),'']
        if s.get('stats'):
            for stat in s['stats']:
                number,label = (stat['number'],stat['label']) if isinstance(stat,dict) else stat
                lines += [f'**{number}:** {label}', '']
        if s.get('table'):
            t=s['table'];lines += ['| '+' | '.join(t['headers'])+' |','| '+' | '.join('---' for _ in t['headers'])+' |']
            lines += ['| '+' | '.join(str(v).replace('\n','<br>') for v in row)+' |' for row in t['rows']];lines += ['']
        if s.get('routes'):
            for route in s['routes']:lines += [f"**{route['name']}:** "+' → '.join(n[0]+' ('+n[1]+')' for n in route['nodes'])]
            lines += ['']
        if s.get('bars'):
            bars=s['bars'];lines += ['**Chart — selected labels:**', '']
            for name,value in zip(bars['labels'],bars['values']):
                number=(f'{value:.4f}' if bars.get('value_format')=='f1' else
                        f'{value:.2%}' if bars.get('value_format')=='percent' else f'{value:,.0f}')
                lines += [f'- {name}: {number}']
            lines += ['']
        if s.get('charts'):
            for chart in s['charts']:lines += [f"**Chart:** {chart['title']} — "+'; '.join(a['name'] for a in chart['series'])+'.']
            lines += ['']
        if s.get('chart'):lines += ['**Chart:** validation Macro-F1 versus decision threshold for TF-IDF, BiGRU and PhoBERT.','']
        if s.get('scatter'):lines += ['**Chart:** training frequency versus validation F1 across all 27 emotion labels.','']
        for item in s.get('annotations',[]):lines += [item['value'],'']
        if s.get('panels'):
            for panel in s['panels']:lines += [f"**{panel['heading']} — {panel['id']}:** {panel['truth']}; {panel['prediction']}; {panel['probability']}. {panel['hypothesis']}"]
            lines += ['']
        if s.get('strategies'):
            lines += [s['checkpoint'],'']+[f'**{a}:** {b}' for a,b in s['strategies']]+['']
            lines += [' → '.join(n[0] for n in s['nodes']),'']
        if s.get('findings'):lines += [f'**{a}:** {b}' for a,b in s['findings']]+['']
        if s.get('limits'):lines += ['**Limitations:** '+' '.join(s['limits']),'']
        if s.get('caption'):lines += [s['caption'],'']
        if s.get('takeaway'):lines += ['**Takeaway:** '+s['takeaway'],'']
        lines += ['**Source / protocol:** '+s['source'],'','**Speaker notes:** '+s['notes'],'']
    return '\n'.join(lines)


def build_presentation():
    from academic_slides import render_deck
    slides=presentation_content()
    if len(slides)!=18:
        raise ValueError('The presentation must contain exactly 18 slides')
    markdown,words=script_markdown(slides)
    if words>950:
        raise ValueError('Shorten the spoken script to fit the presentation time limit')
    write_json(ROOT/'reports/midterm_slide_design.json',slides)
    boxes=render_deck(slides,ROOT/'reports/midterm_slides.pdf')
    (ROOT/'reports/midterm_slides.md').write_text(slide_markdown(slides))
    SCRIPT.write_text(markdown);render_script(markdown)
    tables=sum('table' in slide for slide in slides)
    print(f'Built {len(slides)} academic slides, {tables} tables and a matching script ({words} spoken words).')
    return slides,boxes


if __name__ == '__main__':
    build_presentation()
