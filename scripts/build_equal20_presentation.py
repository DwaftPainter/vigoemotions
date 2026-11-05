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

    add('ViGoEmotions: Vietnamese Emotion Recognition', [
        'Multi-label classification of Vietnamese comments.',
        'Three models; 27 emotions or 28 labels with Neutral.',
        'Both neural models: 20 epochs in each setting.',
        'All reported model scores are validation results.'],
        'Good morning, everyone. Our project is ViGoEmotions, a study of emotion recognition in Vietnamese social-media comments. '
        'A comment may express several emotions at the same time, so this is a multi-label classification task. '
        'We compare three models and two label settings, with and without an explicit Neutral output. '
        'Today, we will explain our data, experiments, main findings, and plan for the final evaluation.')

    add('Dataset and Quality Checks', [
        '20,664 comments; supplied train/validation/test split.',
        'One train–test text overlap removed from training.',
        'Effective sizes: 16,530 / 2,066 / 2,067.',
        'Source annotations: 725 Neutral-only; 306 mixed.'],
        'The dataset contains twenty thousand, six hundred and sixty-four comments. We keep the supplied training, validation, and test partitions. '
        'Our audit found one training comment that duplicated a test comment. We removed the training copy, leaving sixteen thousand, five hundred and thirty training samples. '
        'We also found comments annotated with both Neutral and an emotion. We report this inconsistency and define how each model handles it, rather than silently changing the annotations.')

    add('Emotion Frequency and Imbalance', [
        '27 emotions; nine labels per frequency tier.',
        'Amusement is most frequent; relief is least frequent.',
        'Maximum/minimum training count: about 4.52×.',
        'Positive labels are sparse across the output vector.'],
        'This chart shows label frequencies in the training set. We divide the labels into head, middle, and tail groups, with nine labels in each group. '
        'Amusement is the most frequent emotion, while relief is the least frequent. The ratio is about four point five. '
        'Another challenge is that most positions in a label vector are zero. This is why we focus on F1 and additional multi-label metrics instead of relying on accuracy.',
        'figures/label_frequency.png')

    add('Label Cardinality and Text Length', [
        'Training mean: 1.859 emotions per comment.',
        'Maximum: five emotions in one comment.',
        'Median: 11 whitespace tokens; 47 characters.',
        'Informal language: slang, teencode, emoji and HTML.'],
        'Label cardinality means the number of positive labels in one comment. In training, the average is about one point eight six emotions, and the maximum is five. '
        'The median comment has eleven whitespace tokens and forty-seven characters. A whitespace token is not necessarily a complete Vietnamese word or a PhoBERT token. '
        'The comments also contain slang, teencode, emojis, and non-standard syntax, which make emotion recognition more difficult.',
        'figures/distributions.png')

    add('Emotions Often Occur Together', [
        'Disappointment–sadness: 904 training comments.',
        'Anger–annoyance: 699; sadness–grief: 620.',
        'Co-occurrence and correlation use training labels.',
        'Correlation does not establish a causal relationship.'],
        'Some emotions often appear together. The most frequent pair in our training analysis is disappointment and sadness, with nine hundred and four comments. '
        'We examine both co-occurrence counts and correlation, because a large count may simply reflect frequent individual labels. '
        'These statistics are calculated from training data. Later, we check whether the models can predict both emotions together on validation examples.',
        'figures/train_correlation.png')

    add('Split Protocol and Leakage Prevention', [
        'Preserve the supplied split and fixed sample IDs.',
        'Fit vocabulary and IDF on training data only.',
        'Normalize Unicode and whitespace; preserve negation.',
        'Distribution audit does not prove stratification.',
        'No model evaluation or threshold tuning on test.'],
        'We preserve the supplied split and verify sample identities and text overlap. Vocabulary and inverse document frequency are fitted only on training data. '
        'Our shared preprocessing normalizes Unicode and whitespace while preserving Vietnamese accents and negation. '
        'The split audit describes label distributions, but it does not prove which stratification algorithm originally created the split. '
        'That requirement still needs clarification with the lecturer. We have not evaluated model performance or tuned thresholds on the test set.')

    add('Two Label Settings: 27 vs 28', [
        '27 outputs: Neutral-only becomes an all-zero vector.',
        '28 outputs: Neutral is a separate learned label.',
        'Preserve mixed annotations; no exclusion rule.',
        'Validation: 75 Neutral-only + 19 mixed = 94.',
        'Independent sigmoid outputs with binary cross-entropy.'],
        'In the twenty-seven-label setting, the model predicts emotions only. A Neutral-only comment therefore has an all-zero emotion vector. '
        'In the twenty-eight-label setting, Neutral becomes an additional output. We preserve comments with mixed annotations and do not force Neutral to exclude other emotions. '
        'There are ninety-four Neutral-positive validation examples. Both neural settings use independent sigmoid outputs and binary cross-entropy, because multiple labels can be positive together.')

    add('Three Models, One Experimental Profile', [
        'TF-IDF + OvR LR: liblinear, max_iter = 1,000.',
        'BiGRU: embedding/hidden 128; batch 64; lr 0.001.',
        'PhoBERT: full fine-tuning; batch 16; lr 2e-5.',
        'BiGRU and PhoBERT: 20 epochs for both label sets.',
        'TF-IDF uses a solver limit, not neural epochs.'],
        'Our first model combines TF-IDF with one-versus-rest Logistic Regression. It is a simple lexical baseline. '
        'The second model is a bidirectional GRU. It reads the sequence in both directions and learns its embeddings from scratch. '
        'The third is PhoBERT, a pretrained Vietnamese transformer that we fine-tune, including its backbone. '
        'Both neural models train for twenty epochs in each label setting. Logistic Regression has a solver iteration limit instead, so we do not describe it as a twenty-epoch neural model.')

    add('Checkpoint Selection and Reproducibility', [
        'Seed 42; threshold 0.5; same split and settings.',
        'Select by validation Macro-F1 on the 27 emotions.',
        f'Best BiGRU epochs, 27/28: {int(g27.best_epoch)} / {int(g28.best_epoch)}.',
        f'Best PhoBERT epochs, 27/28: {int(p27.best_epoch)} / {int(p28.best_epoch)}.',
        'Same GPU model and recorded package versions.'],
        'For a clearer comparison, both settings use the same seed, split, and configuration for each model. '
        'We select neural checkpoints using validation Macro-F1 on the same twenty-seven emotions, at threshold zero point five. Ties keep the earliest epoch. '
        f'The best PhoBERT checkpoints are at epochs {int(p27.best_epoch)} and {int(p28.best_epoch)}, but both runs still complete all twenty epochs. '
        'We record the environment and verify data and source checksums, including the line-ending differences between the two sessions.')

    add('Results: 27 Emotions', [
        f'TF-IDF Macro-F1: {results.loc[(27,"tfidf_lr")].macro_f1:.4f}.',
        f'BiGRU Macro-F1: {g27.macro_f1:.4f}.',
        f'PhoBERT Macro-F1: {p27.macro_f1:.4f}; Micro-F1: {p27.micro_f1:.4f}.',
        f'PhoBERT Hamming Loss: {p27.hamming_loss:.4f}; LRAP: {p27.lrap:.4f}.',
        'Higher F1/LRAP is better; lower Hamming Loss is better.'],
        f'For twenty-seven emotions, Macro-F1 is {results.loc[(27,"tfidf_lr")].macro_f1:.4f} for TF-IDF, {g27.macro_f1:.4f} for BiGRU, and {p27.macro_f1:.4f} for PhoBERT. '
        'PhoBERT performs best on all four overall metrics in this setting. Macro-F1 gives equal weight to each label, while Micro-F1 pools decisions across labels. '
        'Hamming Loss measures incorrect label decisions, and LRAP evaluates probability rankings. '
        'PhoBERT training took about four hours on a Tesla T4. The report includes parameter counts, training times, and per-label scores for every model.')

    add('Results: 28 Labels Including Neutral', [
        f'TF-IDF Macro-F1: {results.loc[(28,"tfidf_lr")].macro_f1:.4f}.',
        f'BiGRU Macro-F1: {g28.macro_f1:.4f}.',
        f'PhoBERT Macro-F1: {p28.macro_f1:.4f}; Micro-F1: {p28.micro_f1:.4f}.',
        f'PhoBERT Hamming Loss: {p28.hamming_loss:.4f}; LRAP: {p28.lrap:.4f}.',
        'These metrics include the additional Neutral label.'],
        f'With twenty-eight labels, Macro-F1 is {results.loc[(28,"tfidf_lr")].macro_f1:.4f} for TF-IDF, {g28.macro_f1:.4f} for BiGRU, and {p28.macro_f1:.4f} for PhoBERT. '
        'PhoBERT again has the strongest overall results within this setting. However, these scores include Neutral. '
        'We should not directly subtract the overall twenty-seven-label and twenty-eight-label Macro-F1 values to measure the effect of adding Neutral, because the evaluated label sets are different.')

    add('Does Adding Neutral Help the 27 Emotions?', [
        f'PhoBERT with 27 outputs: {p27.macro_f1:.4f}.',
        f'PhoBERT with 28 outputs, scored on 27: {p28.emotion27_macro_f1:.4f}.',
        f'Difference: {delta:+.2f} percentage points.',
        'No observed improvement on the shared 27 emotions.',
        'One seed; no causal or statistical conclusion.'],
        'To compare the same task, we remove the Neutral column from the twenty-eight-output predictions and recalculate the metrics on the twenty-seven emotions. '
        f'PhoBERT then scores {p28.emotion27_macro_f1:.4f}, compared with {p27.macro_f1:.4f} for the twenty-seven-output model. '
        f'This is about {abs(delta):.2f} percentage points lower. TF-IDF is unchanged, and BiGRU also decreases slightly. '
        'This run does not show an improvement from adding Neutral on the shared emotions. One seed is not enough for a causal claim or a statistical significance claim.')

    add('Neutral Detection Remains Difficult', [
        f'PhoBERT Neutral F1: {neutral.f1:.4f}.',
        f'Precision: {neutral.precision:.4f}; recall: {neutral.recall:.4f}.',
        f'{int(neutral.TP)} TP / {int(neutral.FP)} FP / {int(neutral.FN)} FN; support: {int(neutral.support)}.',
        'No Neutral output exists in the 27-label model.',
        'An empty prediction is not automatically Neutral.'],
        'The additional output does allow the model to predict Neutral directly, but this remains difficult. '
        f'PhoBERT finds {int(neutral.TP)} of the {int(neutral.support)} true Neutral examples, misses {int(neutral.FN)}, and produces {int(neutral.FP)} false positives. '
        f'Its Neutral F1 is {neutral.f1:.4f}. '
        'The twenty-seven-output model has no Neutral classifier, so its Neutral F1 is undefined rather than zero. Also, an empty prediction should not automatically be interpreted as Neutral.')

    add('Learning Curves Across 20 Epochs', [
        'All four neural runs completed epochs 1–20.',
        'The best checkpoint can occur before the last epoch.',
        'PhoBERT 27 starts at F1 = 0, then improves.',
        'Training loss alone cannot confirm overfitting.'],
        'These curves show training loss and the shared checkpoint-selection metric across all twenty epochs. '
        'An interesting example is PhoBERT with twenty-seven outputs: its first-epoch F1 is zero, but it improves substantially later. '
        'This shows why we should not judge the entire run from the first epoch. After the best checkpoint, validation F1 fluctuates while training loss continues to decrease. '
        'We do not have validation loss, so we avoid claiming confirmed overfitting from these curves alone.',
        figures+'learning_curves.png')

    add('Q1–Q2: Per-Label F1 and Frequency', [
        f'PhoBERT best label: {extremes.highest}, F1 {extremes.highest_f1:.4f}.',
        f'Lowest label: {extremes.lowest}, F1 {extremes.lowest_f1:.4f}.',
        f'Frequency vs F1: Spearman rho {frequency.spearman_rho:.3f}.',
        f'p-value {frequency.p_value:.3f}; descriptive evidence only.',
        'Full per-label results are included in the report.'],
        'The first two midterm questions concern individual labels and training frequency. '
        f'For PhoBERT with twenty-seven outputs, {extremes.highest} has the highest F1, approximately {extremes.highest_f1:.3f}, while {extremes.lowest} has the lowest, approximately {extremes.lowest_f1:.3f}. '
        f'The Spearman correlation between training frequency and F1 is only {frequency.spearman_rho:.3f}, with a p-value of {frequency.p_value:.3f}. '
        'These results do not support a simple rule that more frequent emotions are always easier to predict. They are descriptive results from this run.')

    add('Q3–Q4: Label Pairs and Text Styles', [
        f'Disappointment–sadness: PhoBERT gets both in {int(pair.joint_TP)}/{int(pair.val_true)} cases.',
        f'Joint recall at 0.5: {pair.joint_recall*100:.2f}%.',
        f'Largest PhoBERT–BiGRU style gap: {style_name}.',
        f'Macro-F1 gap in this group: +{style.delta_pp:.2f} percentage points.',
        'Style groups use heuristics and may overlap.'],
        'For label dependencies, there are one hundred and twenty-two validation comments containing both disappointment and sadness. '
        f'PhoBERT predicts both labels correctly in {int(pair.joint_TP)} cases, giving a joint recall of about {pair.joint_recall*100:.1f} percent. '
        f'For text styles, the largest PhoBERT advantage over BiGRU is in {style_name}, at about {style.delta_pp:.1f} percentage points of Macro-F1. '
        'Teencode refers to informal Vietnamese online spelling and abbreviations. Our style groups are heuristic and can overlap, so we interpret this comparison cautiously.')

    add('Q5: Sensitivity to the Decision Threshold', [
        f'PhoBERT 27 at 0.5: Macro-F1 {threshold.default_macro:.4f}.',
        f'Best validation grid point: {threshold.best_exploratory_threshold:.2f}, F1 {threshold.exploratory_macro:.4f}.',
        'Same checkpoint; exploratory validation analysis.',
        'Very small F1 gain; Hamming Loss increases.',
        'Main results remain at threshold 0.5.'],
        'Our fifth question examines the default decision threshold. We keep the checkpoint fixed and vary the threshold on validation predictions. '
        f'For PhoBERT, the best point in our grid is {threshold.best_exploratory_threshold:.2f}, with Macro-F1 {threshold.exploratory_macro:.4f}, compared with {threshold.default_macro:.4f} at zero point five. '
        'The gain is very small, while Hamming Loss becomes worse. This is exploratory validation analysis, not a test improvement. '
        'We keep zero point five for the main results and use this observation to motivate our final research question.')

    add('Error Analysis: New Neutral Predictions', [
        'Six PhoBERT 28 cases: three Neutral FP and three FN.',
        'Inspect implied emotion, short text, HTML and profanity.',
        'Publish IDs and abstract observations, not raw comments.',
        'Preliminary AI-assisted review; team confirmation pending.',
        'Error candidates are not a completed manual audit.'],
        'A preliminary, AI-assisted review examines six cases from the new twenty-eight-label model: three confident Neutral false positives and three Neutral false negatives. '
        'These cases suggest difficulties with implied emotions, short expressions, HTML, and profanity. These are hypotheses from selected examples, not proven causes. '
        'The review still needs team confirmation. We retain sample IDs and abstract descriptions without reproducing the original comments, and we distinguish automatically generated error candidates from a completed manual audit.')

    add('Final Research Question and Next Steps', [
        'Is one threshold suitable for every emotion?',
        'Freeze the best neural checkpoint; keep weights fixed.',
        'Compare 0.5, global tuned and per-label thresholds.',
        'Tune on validation; lock choices; evaluate test once.',
        'Review at least 20 false negatives and 20 false positives.'],
        'For the final evaluation, we will ask whether one threshold is suitable for every emotion. '
        'We will freeze the strongest neural checkpoint and compare three decoding strategies: the fixed threshold of zero point five, one tuned global threshold, and separate thresholds for individual labels. '
        'Thresholds will be selected on validation data. Only after locking all choices will we evaluate the held-out test set. '
        'We also plan a more detailed review of at least twenty false negatives and twenty false positives.')

    add('Conclusions and Evidence', [
        'PhoBERT leads within both label settings.',
        'Adding Neutral does not improve shared-emotion F1 in this run.',
        'Evidence: notebooks 08/09, saved predictions and manifests.',
        'Limits: one seed, validation selection, no test scores.',
        'Sources: course brief, ViGoEmotions, PhoBERT and libraries.'],
        'Our main finding is that PhoBERT achieves the strongest overall results within both label settings. '
        'Adding an explicit Neutral output does not improve the shared twenty-seven-emotion F1 in this run, and Neutral detection still has substantial errors. '
        'The results are supported by saved Kaggle predictions, training histories, and provenance checks. '
        'Our current limitations are one seed, validation-based selection, and no model test scores. Thank you for listening. We are happy to take questions.')
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
        f'20 slides; {words:,} spoken words. Target: about 12–15 minutes at a measured pace, including short pauses and handovers. Q&A is additional.', '',
        'Use this script with reports/midterm_slides.pdf. The numerical results come from the verified Kaggle 20-epoch studies. The Vietnamese report and study guide remain available for preparation.', '',
        '## Suggested timing', '',
        '- Slides 1–6: data and preprocessing, about 3–4 minutes.',
        '- Slides 7–9: label settings and methods, about 2 minutes.',
        '- Slides 10–14: results and learning curves, about 3–4 minutes.',
        '- Slides 15–20: analysis, final research question and conclusion, about 4–5 minutes.', '',
        '## Before presenting', '',
        '- Add your real group details to the submitted materials; do not read these preparation instructions aloud.',
        '- Confirm the six AI-assisted error-review observations before presenting them as your team’s analysis.',
        '- Rehearse once with a timer. Read key results, rather than every number on a slide.',
        '- Possible handover points: after slide 6 (data), slide 9 (methods), and slide 14 (results). Allocate these sections to your actual team members.', '',
        '## Slide-by-slide script', '']
    for i, (title, _, _, speech) in enumerate(slides, 1):
        lines.extend([f'### Slide {i:02d} — {title}', '', speech, ''])
        if i in (6, 9, 14):
            transition = {6: 'That completes our data overview. My teammate will now explain the model settings.',
                          9: 'We have defined the experimental protocol. My teammate will now present the results.',
                          14: 'We have seen the main results. My teammate will now discuss the midterm questions and next steps.'}[i]
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
    from build_report import register_fonts, render_slides
    slides = presentation_content()
    write_json(OUT/'slides.json', slides)
    register_fonts(); render_slides(slides, language='en')
    markdown, words = script_markdown(slides)
    SCRIPT.write_text(markdown)
    render_script(markdown)
    print(f'Built 20 English slides and an English script ({words} spoken words).')


if __name__ == '__main__':
    build_presentation()
