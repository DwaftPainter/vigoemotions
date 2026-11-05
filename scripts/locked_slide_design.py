"""Presentation styling only: 15 approved slides followed by two optional tables."""
import hashlib
import json

import pandas as pd

from import_equal20_results import OUT

APPROVED_SHA256 = 'b3f255f12caeb8c40c030f54cde755dfb8f8cc8f2b73f07a9be2a9baac133335'
SUPPLEMENT_NOTE = '''
## Supplementary tables — slides 16–17

Slides 1–15 are the approved main presentation. Their wording, order, numerical results and speaker notes are unchanged. Keep the original timing and handovers above.

Slides 16–17 are optional backup tables for Q&A; they are not part of the timed script. Slide 16 contains the six-model validation results, with 27-label and 28-label metrics in separate blocks. Slide 17 gives additional pair-preservation and text-style numbers from the 27-output study. Show these only if useful when answering a question; no extra spoken script is required.
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
                   layout=layout, approved_slide=i, supplemental=False)
              for i, ((title, bullets, figure, notes), layout) in enumerate(zip(approved, layouts), 1)]

    comparison = pd.read_csv(OUT / 'model_comparison.csv')
    if not (len(comparison) == 6 and set(comparison.label_count) == {27, 28}
            and comparison.split.eq('val').all() and comparison.threshold.eq(.5).all()
            and comparison[comparison.model != 'tfidf_lr'].epochs_trained.eq(20).all()):
        raise ValueError('Appendix tables require the six verified equal-20 validation runs')
    names = {'tfidf_lr': 'TF-IDF + OvR LR', 'gru': 'BiGRU', 'phobert': 'PhoBERT'}
    rows, highlights = [], []
    for labels in (27, 28):
        for model, name in names.items():
            row = comparison[(comparison.label_count == labels) & (comparison.model == model)].iloc[0]
            seconds = row.train_seconds
            time = f'{seconds:.2f} s' if seconds < 10 else f'{seconds:.1f} s' if seconds < 120 else f'{seconds/3600:.2f} h'
            if model == 'phobert':
                highlights.extend((len(rows), j) for j in range(1, 6))
            rows.append([str(labels), name, *[f'{row[m]:.4f}' for m in ('macro_f1', 'micro_f1', 'hamming_loss', 'lrap')],
                         '—' if pd.isna(row.best_epoch) else str(int(row.best_epoch)), time])
    design.append(dict(title='Supplementary Table A · Model performance', layout='appendix_performance',
        supplemental=True, notes='Optional backup for Q&A. Main presentation ends at slide 15.',
        subtitle='Validation n = 2,066 · threshold 0.5 · full-label scores reported separately',
        tables=[dict(headers=['Labels', 'Model', 'Macro-F1 ↑', 'Micro-F1 ↑', 'Hamming ↓', 'LRAP ↑', 'Best ep.', 'Train time'],
                     rows=rows, x=64, top=520, widths=[85,205,145,145,145,145,110,172], row_height=49,
                     group_starts=[3], highlights=highlights)],
        caption='Both neural models trained for 20 epochs. Best ep. is the selected checkpoint. Rank models within each label block; full 27-label and 28-label Macro-F1 use different label sets.',
        source='Source: current model_comparison.csv. CPU TF-IDF; Tesla T4 neural. Times exclude preprocessing; no test scores.'))

    pair = pd.read_csv(OUT / 'pair_preservation.csv')
    styles = pd.read_csv(OUT / 'neural_style_comparison.csv')
    style_names = {'all': 'All', 'emoji_heavy_ge_2': 'Emoji-heavy', 'short_le_5': 'Short', 'teencode': 'Teencode'}
    design.append(dict(title='Supplementary Table B · Pair and text-style analysis', layout='appendix_analysis',
        supplemental=True, notes='Optional backup for Q&A; groups overlap and findings are descriptive.',
        subtitle='27-output study only · validation · threshold 0.5 · supplementary detail for slide 11',
        tables=[dict(headers=['Model', 'Joint TP', 'Support', 'Recall'],
                     rows=[[r.model, str(r.joint_TP), str(r.val_true), f'{r.joint_recall:.2%}'] for r in pair.itertuples()],
                     x=64, top=470, widths=[215,100,105,120], row_height=65, alignments=[0,2,2,2], highlights=[(2,1),(2,3)]),
                dict(headers=['Group', 'n', 'BiGRU F1', 'PhoBERT F1', 'Δ (pp)'],
                     rows=[[style_names[r.style], f'{r.n:,}', f'{r.gru:.4f}', f'{r.phobert:.4f}', f'{r.delta_pp:+.2f}']
                           for r in styles.itertuples()], x=644, top=470, widths=[150,70,112,125,115], row_height=52,
                     alignments=[0,2,2,2,2], highlights=[(3,4)])],
        caption='Joint recall counts both emotions predicted among examples annotated with both; extra predicted labels are allowed. Style groups overlap and differ in size; Δ = PhoBERT − BiGRU in Macro-F1 percentage points. One seed; no causal or significance claim.',
        source='Sources: current pair_preservation.csv; neural_style_comparison.csv. Pair selected by training co-occurrence; no test metrics.'))
    return design


def presentation_markdown(slides, design):
    """Keep the complete approved Markdown as an identical prefix."""
    lines = [main_markdown(slides), '\n<!-- Optional supplementary slides: 16–17; use for Q&A only. -->\n']
    for i, slide in enumerate(design[15:], 16):
        lines += [f"## {i}. {slide['title']}", '', slide['subtitle'], '']
        for table in slide['tables']:
            lines += ['| ' + ' | '.join(table['headers']) + ' |',
                      '| ' + ' | '.join('---' for _ in table['headers']) + ' |']
            lines += ['| ' + ' | '.join(row) + ' |' for row in table['rows']]
            lines += ['']
        lines += [slide['caption'], '', slide['source'], '', 'Speaker notes: ' + slide['notes'], '']
    return '\n'.join(lines)
