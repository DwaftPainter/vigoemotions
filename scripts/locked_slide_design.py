"""Keep the approved script; replace only slides 10–11 with verified Q1–Q4 tables."""
import hashlib
import json

import pandas as pd

from import_equal20_results import OUT

APPROVED_SHA256 = 'b3f255f12caeb8c40c030f54cde755dfb8f8cc8f2b73f07a9be2a9baac133335'
TABLE_NOTE = '''
## Q1–Q4 result tables — slides 10–11

The presentation contains 15 slides. Slides 10–11 use the Q1–Q4 tables; the other 13 slides and all approved speaker notes remain unchanged. Keep the original timing and handovers above.

Slide 10 contains dedicated tables for Q1 (highest and lowest per-label F1) and Q2 (training-frequency versus F1 correlations). Slide 11 contains dedicated tables for Q3 (joint label recall) and Q4 (text-style Macro-F1). All tables use the 27-output validation study. Follow the existing script for these two slides and read the key PhoBERT results rather than every table entry.
'''


def approved_content(slides):
    """Reject editorial changes against the already committed evidence snapshot."""
    source = OUT / 'slides.json'
    if hashlib.sha256(source.read_bytes()).hexdigest() != APPROVED_SHA256:
        raise ValueError('The approved slide-content snapshot changed')
    approved = json.loads(source.read_text())
    if json.loads(json.dumps(slides)) != approved:
        raise ValueError('Titles, bullets, figures and speaker notes are locked for slides 1–15')
    return approved


def main_markdown(slides):
    lines = ['# ViGoEmotions — Midterm presentation', '']
    for i, (title, bullets, figure, note) in enumerate(slides, 1):
        lines += [f'## {i}. {title}', '', *['- ' + b for b in bullets], '',
                  f'Speaker notes: {note}', '']
    return '\n'.join(lines)


def design_slides(slides):
    approved = approved_content(slides)
    layouts = ['title', 'figure', 'encoding', 'models', 'performance', 'performance',
               'neutral_effect', 'neutral_detection', 'figure', 'questions', 'questions',
               'threshold', 'errors', 'plan', 'conclusion']
    design = [dict(title=title, bullets=bullets, figure=figure, notes=notes,
                   layout=layout, approved_slide=i, research_tables=False)
              for i, ((title, bullets, figure, notes), layout) in enumerate(zip(approved, layouts), 1)]

    extremes = pd.read_csv(OUT / 'label_extremes.csv')
    frequency = pd.read_csv(OUT / 'frequency_f1.csv')
    design.append(dict(title='Q1–Q2: Detailed Result Tables', layout='question_tables',
        research_tables=True, notes='Use with slide 10 or during Q&A. The approved main script remains unchanged.',
        subtitle='27-output study only · validation · threshold 0.5 · dedicated tables for Q1 and Q2',
        tables=[dict(rq='Q1', question_title='Q1 / Highest and lowest per-label F1',
                     headers=['Model', 'Highest label / F1', 'Lowest label / F1'],
                     rows=[[r.model, f'{r.highest}\n{r.highest_f1:.4f}', f'{r.lowest}\n{r.lowest_f1:.4f}'] for r in extremes.itertuples()],
                     x=64, top=470, widths=[150,200,210], row_height=76, alignments=[0,0,0], highlights=[(2,1),(2,2)]),
                dict(rq='Q2', question_title='Q2 / Frequency–F1 correlation',
                     headers=['Model', 'Spearman ρ', 'p-value'],
                     rows=[[r.model, f'{r.spearman_rho:.3f}', f'{r.p_value:.3f}'] for r in frequency.itertuples()],
                     x=664, top=470, widths=[190,180,182], row_height=76, alignments=[0,2,2], highlights=[(2,1)])],
        caption='Q1 reports each model’s highest and lowest validation F1. Q2 correlates training frequency with per-label validation F1 across 27 emotions. Descriptive, one-seed analysis; no causal or significance claim.',
        source='Sources: current label_extremes.csv; frequency_f1.csv. All results use 27 outputs; no test metrics.'))

    pair = pd.read_csv(OUT / 'pair_preservation.csv')
    styles = pd.read_csv(OUT / 'neural_style_comparison.csv')
    style_names = {'all': 'All', 'emoji_heavy_ge_2': 'Emoji-heavy', 'short_le_5': 'Short', 'teencode': 'Teencode'}
    design.append(dict(title='Q3–Q4: Pair and Text-Style Tables', layout='question_tables',
        research_tables=True, notes='Use with slide 11 or during Q&A; groups overlap and findings are descriptive.',
        subtitle='27-output study only · validation · threshold 0.5 · dedicated tables for Q3 and Q4',
        tables=[dict(rq='Q3', question_title='Q3 / Disappointment + sadness · train n = 904', headers=['Model', 'Joint TP', 'Support', 'Recall'],
                     rows=[[r.model, str(r.joint_TP), str(r.val_true), f'{r.joint_recall:.2%}'] for r in pair.itertuples()],
                     x=64, top=470, widths=[215,100,105,120], row_height=65, alignments=[0,2,2,2], highlights=[(2,1),(2,3)]),
                dict(rq='Q4', question_title='Q4 / Text-style Macro-F1', headers=['Group', 'n', 'BiGRU F1', 'PhoBERT F1', 'Δ (pp)'],
                     rows=[[style_names[r.style], f'{r.n:,}', f'{r.gru:.4f}', f'{r.phobert:.4f}', f'{r.delta_pp:+.2f}']
                           for r in styles.itertuples()], x=644, top=470, widths=[150,70,112,125,115], row_height=52,
                     alignments=[0,2,2,2,2], highlights=[(3,4)])],
        caption='Joint recall counts both emotions predicted among examples annotated with both; extra predicted labels are allowed. Style groups overlap and differ in size; Δ = PhoBERT − BiGRU in Macro-F1 percentage points. One seed; no causal or significance claim.',
        source='Sources: current pair_preservation.csv; neural_style_comparison.csv. Pair selected by training co-occurrence; no test metrics.'))
    replacements = design[-2:]
    design = design[:15]
    for index, replacement in zip((9, 10), replacements):
        replacement.update(notes=approved[index][3], approved_slide=index+1, replaces_slide=index+1)
        design[index] = replacement
    return design


def presentation_markdown(slides, design):
    """Only the two explicitly requested table replacements change main content."""
    approved_content(slides)
    lines = ['# ViGoEmotions — Midterm presentation', '']
    for i, slide in enumerate(design, 1):
        if slide['layout'] != 'question_tables':
            title, bullets, figure, note = slides[i-1]
            lines += [f'## {i}. {title}', '', *['- '+b for b in bullets], '', f'Speaker notes: {note}', '']
            continue
        lines += [f"## {i}. {slide['title']}", '', slide['subtitle'], '']
        for table in slide['tables']:
            lines += [f"### {table['question_title']}", '', '| ' + ' | '.join(table['headers']) + ' |',
                      '| ' + ' | '.join('---' for _ in table['headers']) + ' |']
            lines += ['| ' + ' | '.join(cell.replace('\n','<br>') for cell in row) + ' |' for row in table['rows']]
            lines += ['']
        lines += [slide['caption'], '', slide['source'], '', 'Speaker notes: ' + slide['notes'], '']
    return '\n'.join(lines)
