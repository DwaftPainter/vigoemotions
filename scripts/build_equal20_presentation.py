"""English midterm slides and a matching rehearsal script; no model execution."""
import json
import re
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, KeepTogether

from common import ROOT, write_json
from import_equal20_results import OUT

SCRIPT = ROOT/'docs/midterm_presentation_en.md'


def presentation_content():
    comparison = pd.read_csv(OUT/'model_comparison.csv')
    neural = comparison[comparison.model != 'tfidf_lr']
    if len(comparison) != 6 or len(neural) != 4 or not neural.epochs_trained.eq(20).all():
        raise ValueError('The presentation requires the six verified equal-20 results')
    results = comparison.set_index(['label_count', 'model'])
    p27, p28 = results.loc[(27, 'phobert')], results.loc[(28, 'phobert')]
    g27, g28 = results.loc[(27, 'gru')], results.loc[(28, 'gru')]
    neutral = pd.read_csv(OUT/'neutral_metrics.csv').set_index('model').loc['PhoBERT']
    extremes = pd.read_csv(OUT/'label_extremes.csv').set_index('model').loc['PhoBERT']
    frequency = pd.read_csv(OUT/'frequency_f1.csv').set_index('model').loc['PhoBERT']
    pair = pd.read_csv(OUT/'pair_preservation.csv').set_index('model').loc['PhoBERT']
    styles = pd.read_csv(OUT/'neural_style_comparison.csv')
    style = styles[styles['style'] != 'all'].sort_values('delta_pp', ascending=False).iloc[0]
    style_name = {'teencode': 'teencode', 'short_le_5': 'short comments', 'emoji_heavy_ge_2': 'emoji-heavy comments'}[style['style']]
    threshold = pd.read_csv(OUT/'threshold_sensitivity.csv').set_index('model').loc['PhoBERT']
    delta = 100*(p28.emotion27_macro_f1-p27.macro_f1)
    figures = str(OUT.relative_to(ROOT/'reports'))+'/figures/'
    slides = []
    def add(title, bullets, speech, figure=None):
        slides.append((title, bullets, figure, speech))

    add('ViGoEmotions: Task and Dataset', [
        'Multi-label emotion recognition in Vietnamese comments.',
        '20,664 comments; 27 emotions plus a Neutral state.',
        'Effective train / validation / test: 16,530 / 2,066 / 2,067.',
        'Three models; compare 27 and 28 outputs.',
        'All model scores are validation results.'],
        'Good morning. Our project recognizes emotions in Vietnamese social-media comments. '
        'Because one comment can express several emotions, this is a multi-label task. '
        'The dataset contains twenty thousand, six hundred and sixty-four comments. '
        'We compare three models, with and without an explicit Neutral output. I will briefly cover the setup, then focus on our results and analysis.')

    add('Data Profile and Leakage Prevention', [
        'Mean 1.859 emotions; median 11 tokens / 47 characters.',
        'Amusement–relief frequency ratio: about 4.52×.',
        'Fixed split; one overlapping training sample removed.',
        'Fit vocabulary/IDF on train only; no model test scores.'],
        'Comments average about one point eight six emotions, and the median length is eleven whitespace tokens. '
        'We keep the supplied split, remove one training copy of a test comment, and fit vocabulary only on training data. '
        'Unicode and whitespace are normalized. Our distribution audit does not prove the original stratification algorithm, so that requirement still needs clarification.',
        'figures/label_frequency.png')

    add('27 vs 28 Outputs: What Changes?', [
        '27 outputs: Neutral-only has an all-zero emotion target.',
        '28 outputs: Neutral becomes a separate learned label.',
        'Keep mixed annotations; no Neutral exclusion rule.',
        'Validation: 75 Neutral-only + 19 mixed = 94.',
        'Independent sigmoid outputs with binary cross-entropy.'],
        'The twenty-seven-output model predicts emotions only. Neutral-only comments have an all-zero emotion target. '
        'The twenty-eight-output model adds a learned Neutral label. We preserve mixed annotations, where Neutral appears with emotions. '
        'Both neural settings use sigmoid outputs and binary cross-entropy because labels are not mutually exclusive. Validation contains ninety-four Neutral-positive examples.')

    add('Models and Shared Training Protocol', [
        'TF-IDF + OvR LR: liblinear; max_iter = 1,000.',
        'BiGRU from scratch: hidden 128; batch 64; lr 0.001.',
        'PhoBERT full fine-tuning: batch 16; lr 2e-5.',
        'Both neural models: 20 epochs per label setting; seed 42.',
        'Select checkpoints by Macro-F1 on the same 27 emotions.'],
        'We compare TF-IDF with one-versus-rest Logistic Regression, a bidirectional GRU trained from scratch, and full PhoBERT fine-tuning. '
        'Both neural models complete twenty epochs in each setting. TF-IDF uses a solver limit instead. '
        'We share the seed and split, and select checkpoints by validation Macro-F1 on the same twenty-seven emotions at threshold zero point five.')

    add('Results: 27 Emotions', [
        f'TF-IDF Macro-F1: {results.loc[(27,"tfidf_lr")].macro_f1:.4f}.',
        f'BiGRU Macro-F1: {g27.macro_f1:.4f}.',
        f'PhoBERT Macro-F1: {p27.macro_f1:.4f}; Micro-F1: {p27.micro_f1:.4f}.',
        f'PhoBERT Hamming Loss: {p27.hamming_loss:.4f}; LRAP: {p27.lrap:.4f}.',
        'Higher F1/LRAP is better; lower Hamming Loss is better.'],
        f'For twenty-seven emotions, Macro-F1 is {results.loc[(27,"tfidf_lr")].macro_f1:.4f} for TF-IDF, '
        f'{g27.macro_f1:.4f} for BiGRU, and {p27.macro_f1:.4f} for PhoBERT. '
        'PhoBERT leads on all four overall metrics in this setting. '
        'Macro-F1 weights labels equally; Micro-F1 pools decisions across labels. Hamming Loss measures incorrect decisions, while LRAP evaluates probability rankings.')

    add('Results: 28 Labels Including Neutral', [
        f'TF-IDF Macro-F1: {results.loc[(28,"tfidf_lr")].macro_f1:.4f}.',
        f'BiGRU Macro-F1: {g28.macro_f1:.4f}.',
        f'PhoBERT Macro-F1: {p28.macro_f1:.4f}; Micro-F1: {p28.micro_f1:.4f}.',
        f'PhoBERT Hamming Loss: {p28.hamming_loss:.4f}; LRAP: {p28.lrap:.4f}.',
        'These metrics evaluate all 28 labels.'],
        f'For twenty-eight labels, Macro-F1 is {results.loc[(28,"tfidf_lr")].macro_f1:.4f}, '
        f'{g28.macro_f1:.4f}, and {p28.macro_f1:.4f}, respectively. PhoBERT again performs best within this setting. '
        'However, these scores include Neutral. We cannot directly subtract the overall twenty-seven-label and twenty-eight-label scores to measure the benefit of Neutral, because their evaluated label sets differ.')

    add('Does Neutral Help the Shared 27 Emotions?', [
        f'PhoBERT with 27 outputs: {p27.macro_f1:.4f}.',
        f'PhoBERT with 28 outputs, scored on 27: {p28.emotion27_macro_f1:.4f}.',
        f'Difference: {delta:+.2f} percentage points.',
        'No observed improvement on the shared emotions.',
        'One seed; no causal or statistical conclusion.'],
        'For a comparison on identical labels, we score only the twenty-seven emotion columns of the expanded model. '
        f'PhoBERT then reaches {p28.emotion27_macro_f1:.4f}, about {abs(delta):.2f} percentage points below the twenty-seven-output model. '
        'TF-IDF is unchanged and BiGRU decreases slightly. This run shows no improvement on the shared emotions, but one seed cannot establish a general causal conclusion.')

    add('Neutral Detection Remains Difficult', [
        f'PhoBERT Neutral F1: {neutral.f1:.4f}.',
        f'Precision: {neutral.precision:.4f}; recall: {neutral.recall:.4f}.',
        f'{int(neutral.TP)} TP / {int(neutral.FP)} FP / {int(neutral.FN)} FN; support: {int(neutral.support)}.',
        'The 27-output model has no Neutral classifier.',
        'An empty prediction is not automatically Neutral.'],
        f'PhoBERT identifies {int(neutral.TP)} of the ninety-four Neutral-positive validation examples. '
        f'It misses {int(neutral.FN)} and produces {int(neutral.FP)} false positives, giving Neutral F1 of {neutral.f1:.4f}. '
        'So the additional label is learned, but still difficult. The twenty-seven-output model has no Neutral classifier; its Neutral F1 is undefined, not zero. An empty prediction is also not automatically Neutral.')

    add('Learning Curves and Best Checkpoints', [
        'All four neural runs completed epochs 1–20.',
        f'Best BiGRU epochs, 27/28: {int(g27.best_epoch)} / {int(g28.best_epoch)}.',
        f'Best PhoBERT epochs, 27/28: {int(p27.best_epoch)} / {int(p28.best_epoch)}.',
        'Select the best checkpoint, not necessarily the last.',
        'Training loss alone cannot confirm overfitting.'],
        'These curves cover all twenty epochs. The best PhoBERT checkpoints occur at epochs fourteen and fifteen. '
        'We finish training before loading the selected checkpoint. Validation F1 fluctuates after the best epoch even while training loss decreases. '
        'Because validation loss was not recorded, we avoid diagnosing confirmed overfitting from training loss alone.',
        figures+'learning_curves.png')

    add('Q1–Q2: Per-Label Performance and Frequency', [
        f'PhoBERT best label: {extremes.highest}, F1 {extremes.highest_f1:.4f}.',
        f'Lowest label: {extremes.lowest}, F1 {extremes.lowest_f1:.4f}.',
        f'Training frequency vs F1: Spearman rho {frequency.spearman_rho:.3f}.',
        f'p-value {frequency.p_value:.3f}; descriptive evidence only.',
        'Complete per-label results are in the report.'],
        'For individual emotions, PhoBERT performs best on gratitude and worst on disapproval. '
        f'The correlation between training frequency and per-label F1 is approximately {frequency.spearman_rho:.3f}, '
        f'with a p-value of {frequency.p_value:.3f}. '
        'These descriptive results do not support a simple rule that more frequent labels are always easier. The report includes every label for all three models.')

    add('Q3–Q4: Label Pairs and Text Styles', [
        'Top training pair: disappointment–sadness (904 cases).',
        f'Both correct: {int(pair.joint_TP)}/{int(pair.val_true)} validation cases; joint recall {pair.joint_recall*100:.2f}%.',
        f'Largest PhoBERT–BiGRU style gap: {style_name}.',
        f'Macro-F1 gap: +{style.delta_pp:.2f} percentage points.',
        'Style groups are heuristic and may overlap.'],
        'Our most frequent training pair is disappointment and sadness. '
        f'PhoBERT predicts both correctly in {int(pair.joint_TP)} of '
        f'{int(pair.val_true)} validation cases containing both labels. '
        f'For text styles, its largest advantage over BiGRU is in {style_name}, at about {style.delta_pp:.1f} percentage points of Macro-F1. '
        'Teencode means informal online spellings and abbreviations. Our style groups use heuristics and can overlap, so this finding needs cautious interpretation.')

    add('Q5: Does the Default Threshold Matter?', [
        f'PhoBERT 27 at 0.5: Macro-F1 {threshold.default_macro:.4f}.',
        f'Best validation grid point: {threshold.best_exploratory_threshold:.2f}, F1 {threshold.exploratory_macro:.4f}.',
        'Same checkpoint; exploratory validation analysis.',
        'Very small F1 gain; Hamming Loss increases.',
        'Main results remain at threshold 0.5.'],
        'We also vary the threshold while keeping the checkpoint fixed. '
        f'For PhoBERT, the best validation grid point is {threshold.best_exploratory_threshold:.2f}, '
        f'with Macro-F1 {threshold.exploratory_macro:.4f}, compared with {threshold.default_macro:.4f} at zero point five. '
        'The gain is tiny and Hamming Loss worsens. This is exploratory analysis, not a test improvement; the main comparison stays at zero point five.')

    add('Error Analysis: New Neutral Predictions', [
        'Six cases: three Neutral false positives and three misses.',
        'Potential issues: implied emotion, short text, HTML, profanity.',
        'IDs and abstract observations; no raw comments.',
        'Preliminary AI-assisted review; team confirmation pending.'],
        'A preliminary, AI-assisted review examines three confident Neutral false positives and three Neutral misses. '
        'These examples suggest difficulties with implied emotion, short text, HTML, and profanity. These are hypotheses, not proven causes. '
        'The review needs team confirmation. We publish sample IDs and abstract observations instead of the original comments.')

    add('Final Research Question and Next Steps', [
        'Is one threshold suitable for every emotion?',
        'Freeze the best neural checkpoint and its weights.',
        'Compare 0.5, global tuned and per-label thresholds.',
        'Tune on validation; lock choices; evaluate test once.',
        'Review at least 20 false negatives and 20 false positives.'],
        'For the final evaluation, we will test whether one threshold suits every emotion. '
        'We will freeze the best neural checkpoint and compare the default threshold, a tuned global threshold, and per-label thresholds. '
        'We select thresholds on validation, lock all choices, then evaluate test once. We also plan a detailed review of at least twenty false negatives and twenty false positives.')

    add('Conclusions and Evidence', [
        'PhoBERT leads within both label settings.',
        'Neutral does not improve shared-emotion F1 in this run.',
        'Evidence: Kaggle 08/09, saved predictions and manifests.',
        'Limits: one seed, validation selection, no test scores.',
        'Sources: course brief, ViGoEmotions, PhoBERT and libraries.'],
        'To conclude, PhoBERT gives the strongest overall results in both settings. '
        'Adding Neutral does not improve shared-emotion F1 in this run, and Neutral prediction still makes substantial errors. '
        'Saved Kaggle predictions and histories support these findings. Our next step is controlled threshold evaluation. Thank you; we welcome your questions.')
    return slides


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
    words = sum(len(re.findall(r"\b[\w'-]+\b", speech)) for _, _, _, speech in slides)
    lines = ['# ViGoEmotions — English Midterm Presentation Script', '',
        f'{len(slides)} slides; {words:,} spoken words. Target: about 8–9 minutes, including short pauses and handovers. Finish before the 10-minute limit; leave any remaining time for questions.', '',
        'Use this script with reports/midterm_slides.pdf. The numerical results come from the verified Kaggle 20-epoch studies. The Vietnamese report and study guide remain available for preparation.', '',
        '## Suggested timing', '',
        '- Slides 1–4: compact introduction, data and protocol, about 2 minutes.',
        '- Slides 5–9: results and learning curves, about 3 minutes.',
        '- Slides 10–13: midterm questions and errors, about 2–2.5 minutes.',
        '- Slides 14–15: final research question and conclusion, about 1 minute.', '',
        '## Before presenting', '',
        '- Add your real group details to the submitted materials; do not read these preparation instructions aloud.',
        '- Confirm the six AI-assisted error-review observations before presenting them as your team’s analysis.',
        '- Rehearse once with a timer. Read key results, rather than every number on a slide.',
        '- Possible handover points: after slide 4 (setup) and slide 9 (results). Allocate these sections to your actual team members.',
        '- Rehearse at roughly 110–120 words per minute. If behind schedule, shorten the learning-curve and frequency comments; keep the main results and conclusion.', '',
        '## Slide-by-slide script', '']
    for i, (title, _, _, speech) in enumerate(slides, 1):
        lines.extend([f'### Slide {i:02d} — {title}', '', speech, ''])
        if i in (4, 9):
            transition = {4: 'That is our setup. My teammate will now present the results.',
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


def build_presentation():
    from academic_slides import render_deck
    from locked_slide_design import design_slides, presentation_markdown, SUPPLEMENT_NOTE
    slides = presentation_content()
    if len(slides) != 15:
        raise ValueError('The short presentation must contain exactly 15 slides')
    design = design_slides(slides)
    write_json(ROOT/'reports/midterm_slide_design.json', design)
    boxes = render_deck(design, ROOT/'reports/midterm_slides.pdf')
    (ROOT/'reports/midterm_slides.md').write_text(presentation_markdown(slides, design))
    markdown, words = script_markdown(slides)
    if words > 950:
        raise ValueError('Shorten the spoken script to fit the presentation time limit')
    SCRIPT.write_text(markdown + SUPPLEMENT_NOTE)
    render_script(markdown + SUPPLEMENT_NOTE)
    print(f'Built {len(slides)} unchanged main slides + 2 optional appendix tables; {words} unchanged spoken words.')
    return design, boxes


if __name__ == '__main__':
    build_presentation()
