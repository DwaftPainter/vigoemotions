"""Build the current midterm exclusively from verified equal-20 Kaggle results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, LABELS, load_data, targets, write_json
from import_equal20_results import OUT, MODELS

NAMES = {'tfidf_lr': 'TF-IDF + OvR LR', 'gru': 'BiGRU', 'phobert': 'PhoBERT'}
METRICS = ['macro_f1', 'micro_f1', 'hamming_loss', 'lrap']


def run(labels, model):
    return json.loads((OUT/f'evidence/{labels}/{model}/run.json').read_text())


def diagnostic(labels, model, suffix):
    return pd.read_csv(OUT/f'analysis/{labels}_emotion27/{model}/tables/{model}_{suffix}.csv')


def figures(comparison):
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, labels in zip(axes, (27, 28)):
        rows = comparison[comparison.label_count == labels]
        ax.bar([NAMES[m] for m in rows.model], rows.macro_f1, color=['#6497a5', '#e7a140', '#087e8b'])
        for i, v in enumerate(rows.macro_f1):
            ax.text(i, v+.012, f'{v:.4f}', ha='center', fontsize=10)
        ax.set(title=f'{labels} labels; validation; threshold 0.5', ylim=(0, .7), ylabel='Macro-F1')
        ax.tick_params(axis='x', labelsize=9)
    fig.tight_layout(); fig.savefig(folder/'model_metrics.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for column, model in enumerate(('gru', 'phobert')):
        for labels, color in ((27, '#087e8b'), (28, '#d46b36')):
            history = run(labels, model)['history']
            x = [h['epoch'] for h in history]
            axes[0, column].plot(x, [h['loss'] for h in history], label=f'{labels} labels', color=color)
            key = 'macro_f1' if labels == 27 else 'selection_macro_f1'
            axes[1, column].plot(x, [h[key] for h in history], label=f'{labels} labels', color=color)
        axes[0, column].set(title=NAMES[model], ylabel='Training BCE')
        axes[1, column].set(xlabel='Epoch', ylabel='Validation emotion27 Macro-F1')
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(folder/'learning_curves.png', dpi=160); plt.close(fig)


def error_review(frames):
    """Abstract observations from reading six new-study cases; never quote source text."""
    descriptions = {
        '30099': ('Lời giải thích tình huống được gán relief; model chỉ dự đoán Neutral.', 'Cảm xúc hàm ý trong giải thích tình huống có thể khó nhận diện.'),
        'you003230': ('Nội dung liệt kê được gán approval và realization; model chỉ dự đoán Neutral.', 'Dạng liệt kê và HTML có thể che tín hiệu cảm xúc; cần ablation để kiểm chứng.'),
        '15569': ('Lời bình ngắn được gán surprise; model chỉ dự đoán Neutral.', 'Sắc thái ngạc nhiên phụ thuộc tiểu từ/ngữ cảnh ngắn.'),
        '6238': ('Ground truth chỉ Neutral; model dự đoán anger và annoyance.', 'Từ tục và khẩu ngữ có thể kéo dự đoán sang cảm xúc tiêu cực.'),
        '21975': ('Ground truth chỉ Neutral; model dự đoán anger.', 'Tên riêng hoặc cụm từ theo ngữ cảnh có thể gây nhầm; chưa xác định cơ chế.'),
        '1134': ('Ground truth chỉ Neutral; model dự đoán amusement và joy.', 'Từ mang nghĩa ưa thích có thể bị đọc thành cảm xúc tích cực.'),
    }
    path = OUT/'evidence/28/phobert/val_predictions.npz'
    with np.load(path, allow_pickle=False) as a:
        y, p = a['true_label_vector'], a['probability_vector']
        ids = a['sample_id']; pred = p >= .5
    rows = []
    for kind, mask, reverse in [('Neutral FP', (y[:, 27] == 0)&pred[:, 27], True),
                                 ('Neutral FN', (y[:, 27] == 1)&~pred[:, 27], False)]:
        selected = sorted(np.flatnonzero(mask), key=lambda i: float(p[i, 27]), reverse=reverse)[:3]
        for i in selected:
            sample_id = str(ids[i])
            if sample_id not in descriptions:
                raise ValueError('New error candidates require a fresh qualitative review')
            observation, hypothesis = descriptions[sample_id]
            rows.append(dict(sample_id=sample_id, case_type=kind, neutral_probability=float(p[i, 27]),
                true_labels=';'.join(np.array(LABELS+['neutral'])[y[i].astype(bool)]),
                predicted_labels=';'.join(np.array(LABELS+['neutral'])[pred[i]]),
                observation=observation, hypothesis=hypothesis,
                review_status='AI-assisted; pending team confirmation'))
    review = pd.DataFrame(rows)
    review.to_csv(OUT/'phobert28_error_review.csv', index=False)
    return review


def main():
    from build_report import register_fonts, render_report, table
    provenance = json.loads((OUT/'provenance.json').read_text())
    if provenance['historical_results_used'] or provenance['trained_locally']:
        raise ValueError('The midterm must use the verified Kaggle runs')
    comparison = pd.read_csv(OUT/'model_comparison.csv')
    if len(comparison) != 6 or set(comparison.label_count) != {27, 28}:
        raise ValueError('Import and verify both complete studies before building')
    if not comparison[comparison.model != 'tfidf_lr'].epochs_trained.eq(20).all():
        raise ValueError('Current results must use 20 neural epochs')
    per = pd.read_csv(OUT/'per_label_metrics.csv')
    frames, excluded = load_data()
    freq = pd.read_csv(ROOT/'reports/tables/label_frequency.csv')
    np.testing.assert_array_equal(freq.train_count, targets(frames['train']).sum(axis=0))
    for labels in (27, 28):
        manifest = json.loads((OUT/f'evidence/{labels}/manifest.json').read_text())
        for split, checksum in manifest['split_ids_sha256'].items():
            from import_equal20_results import digest
            if digest('\n'.join(frames[split].id).encode()) != checksum:
                raise ValueError('Local data no longer matches evidence')
    figures(comparison)
    review = error_review(frames)
    selected = comparison.set_index(['label_count', 'model'])
    paired = []
    for model in MODELS:
        a, b = selected.loc[(27, model)], selected.loc[(28, model)]
        paired.append(dict(model=NAMES[model], macro27=a.macro_f1,
            macro28_projected27=b.emotion27_macro_f1,
            delta_pp=100*(b.emotion27_macro_f1-a.macro_f1),
            hamming27=a.hamming_loss, hamming28_projected27=b.emotion27_hamming_loss))
    paired = pd.DataFrame(paired); paired.to_csv(OUT/'paired_emotion27_delta.csv', index=False)
    neutral_rows = []
    for model in MODELS:
        with np.load(OUT/f'evidence/28/{model}/val_predictions.npz', allow_pickle=False) as a:
            truth, pred = a['true_label_vector'][:, 27].astype(bool), a['probability_vector'][:, 27] >= .5
        r = per[(per.label_count == 28)&(per.model == model)&(per.label == 'neutral')].iloc[0]
        neutral_rows.append(dict(model=NAMES[model], precision=r.precision, recall=r.recall, f1=r.f1,
            support=int(truth.sum()), TP=int((truth&pred).sum()), FP=int((~truth&pred).sum()), FN=int((truth&~pred).sum())))
    neutral = pd.DataFrame(neutral_rows); neutral.to_csv(OUT/'neutral_metrics.csv', index=False)
    q1, q2, style_rows, pair_rows, sensitivity = [], [], [], [], []
    for model in MODELS:
        rows = per[(per.label_count == 27)&(per.model == model)]
        q1.append(dict(model=NAMES[model], highest='; '.join(rows.loc[rows.f1 == rows.f1.max(), 'label']),
                       highest_f1=rows.f1.max(), lowest='; '.join(rows.loc[rows.f1 == rows.f1.min(), 'label']), lowest_f1=rows.f1.min()))
        analysis = json.loads((OUT/f'analysis/27_emotion27/{model}/{model}_analysis.json').read_text())
        q2.append(dict(model=NAMES[model], spearman_rho=analysis['frequency_f1_spearman'], p_value=analysis['spearman_pvalue']))
        pairs = diagnostic(27, model, 'pair_preservation_val')
        pair = pairs[(pairs.label_a == 'disappointment')&(pairs.label_b == 'sadness')].iloc[0]
        pair_rows.append(dict(model=NAMES[model], val_true=int(pair.val_true_count), joint_TP=int(pair.joint_true_positive), joint_recall=pair.joint_recall))
        sweep = diagnostic(27, model, 'thresholds_val')
        best = sweep.sort_values(['macro_f1', 'threshold'], ascending=[False, False]).iloc[0]
        sensitivity.append(dict(model=NAMES[model], default_macro=selected.loc[(27, model)].macro_f1,
            best_exploratory_threshold=best.threshold, exploratory_macro=best.macro_f1, exploratory_hamming=best.hamming_loss))
    for model in ('gru', 'phobert'):
        s = diagnostic(27, model, 'styles').copy(); s['model'] = model; style_rows.append(s)
    styles = pd.concat(style_rows).pivot(index=['style', 'n'], columns='model', values='macro_f1').reset_index()
    styles['delta_pp'] = 100*(styles.phobert-styles.gru)
    styles.to_csv(OUT/'neural_style_comparison.csv', index=False)
    for name, rows in [('label_extremes', q1), ('frequency_f1', q2), ('pair_preservation', pair_rows), ('threshold_sensitivity', sensitivity)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv', index=False)
    top_style = styles[styles['style'] != 'all'].sort_values('delta_pp', ascending=False).iloc[0]
    ph27, ph28 = selected.loc[(27, 'phobert')], selected.loc[(28, 'phobert')]
    ng = neutral[neutral.model == 'PhoBERT'].iloc[0]
    relative_fig = str(OUT.relative_to(ROOT/'reports'))+'/figures/'
    display = comparison.copy(); display['model'] = display.model.map(NAMES)
    model_columns = ['model', 'macro_f1', 'micro_f1', 'hamming_loss', 'lrap']
    cost = display[['model', 'label_count', 'epochs_trained', 'best_epoch', 'parameter_count', 'train_seconds', 'device']].copy()
    cost.parameter_count = cost.parameter_count.astype(int).astype(str)
    for column in ('epochs_trained', 'best_epoch'):
        cost[column] = cost[column].map(lambda value: 'Không áp dụng' if pd.isna(value) else str(int(value)))
    study27 = table(display[display.label_count == 27], model_columns)
    study28 = table(display[display.label_count == 28], model_columns)
    split = json.loads((ROOT/'reports/split_audit/summary.json').read_text())
    text = f'''# ViGoEmotions — Báo cáo giữa kỳ

**Kết quả chính: Kaggle 05/11/2026, notebook 08/09 executed.** Cả bốn lượt neural đã chạy đủ 20 epoch. Sáu kết quả được tính lại từ xác suất validation trong hai ZIP đầu ra. Không dùng kết quả của các lượt ít epoch trong bảng, biểu đồ hoặc kết luận báo cáo này.

**Thông tin nhóm:** Chưa được cung cấp; bổ sung họ tên, MSSV và phân công trong docs/team_information.md trước khi nộp.

## 1. Bài toán và phạm vi

Nhận diện cảm xúc đa nhãn trên bình luận tiếng Việt. Một mẫu có thể có nhiều cảm xúc, nên mỗi nhãn là một quyết định nhị phân độc lập. Nhóm đánh giá ba mô hình trên hai cách mã hóa: 27 cảm xúc và 28 nhãn gồm Neutral. Mọi điểm mô hình trong báo cáo là validation, ngưỡng 0,5, seed 42; test được giữ cho Final.

Nguồn yêu cầu: student-guide.pdf, mục Midterm; 02-vigoemotions-vietnamese-emotion.pdf, mục Midterm A–E. Cần EDA, split/preprocessing hợp lệ, ba mô hình, Macro/Micro-F1, per-label F1, Hamming Loss, LRAP, tham số/thời gian, phân tích lỗi và RQ Final. Bộ slide có 15 trang; code và README tái lập đi cùng kết quả.

## 2. Dữ liệu và EDA

CSV nguồn có 20.664 bình luận: train 16.531, validation 2.066, test 2.067. Loại {len(excluded)} mẫu train trùng văn bản held-out; train thực dùng là {len(frames['train']):,}. Mapping 27 cảm xúc theo đúng thứ tự ID: {', '.join(LABELS)}. Neutral có ID 27.

Phân nhóm head/mid/tail theo thứ hạng tần suất trên train, chín nhãn mỗi nhóm; đây là tier tương đối. Amusement có tần suất lớn nhất, relief ít nhất, chênh lệch khoảng {freq.train_count.max()/freq.train_count.min():.2f} lần. Nhãn dương thưa: cardinality train trung bình {targets(frames['train']).sum(axis=1).mean():.3f}, tối đa 5 cảm xúc. Median độ dài là 11 token khoảng trắng và 47 ký tự. Whitespace token không đồng nghĩa từ tiếng Việt hoặc BPE token.

![Phân bố 27 cảm xúc trên train; cùng dữ liệu với hai study Kaggle.](figures/label_frequency.png)

![Cardinality, token khoảng trắng và ký tự theo split; không dùng để chọn mô hình bằng test.](figures/distributions.png)

Disappointment–sadness đồng xuất hiện 904 lần trên train, anger–annoyance 699 và sadness–grief 620. Ma trận co-occurrence và tương quan phi chỉ xây từ train; tần suất đồng xuất hiện không chứng minh quan hệ nhân quả. Các đặc điểm tiếng Việt cần chú ý gồm teencode, khẩu ngữ, từ tục, emoji, HTML và cú pháp phi chuẩn.

![Tương quan nhãn trên train, giữ thứ tự mapping 27 cảm xúc.](figures/train_correlation.png)

## 3. Split, preprocessing và hai cách mã hóa

Giữ split được cấp, fit vocabulary và IDF chỉ trên train; chuẩn hóa NFC/gom khoảng trắng, giữ dấu và phủ định. TF-IDF dùng regex unigram/bigram, BiGRU dùng lowercase/whitespace, PhoBERT dùng VnCoreNLP rồi BPE. Không thêm luật sửa annotation.

Notebook 07 xác nhận toàn vẹn {split['integrity_status']}, fixed IDs khớp và không còn overlap sau lọc. Lệch prevalence lớn nhất so với train là {split['per_label_drift']['emotion27_val_max_abs_drift_pp']:.4f} điểm phần trăm ở validation và {split['per_label_drift']['emotion27_test_max_abs_drift_pp']:.4f} ở test. Audit phân bố không chứng minh thuật toán stratification gốc; việc chấp nhận official split theo đề còn cần xác nhận với giảng viên. Test chỉ tham gia audit, không chấm mô hình/chọn ngưỡng.

Với 27 đầu ra, [27] thành vector cảm xúc toàn 0; [2,27] giữ joy. Với 28 đầu ra, [27] là Neutral dương; [2,27] giữ cả joy và Neutral. CSV nguồn có 725 Neutral-only và 306 mixed-neutral; validation có 75 Neutral-only + 19 mixed = 94 Neutral dương. Giữ annotation mixed và công bố bất nhất; không dùng softmax hoặc ép Neutral loại trừ cảm xúc. Dự đoán rỗng không tự động đồng nghĩa Neutral.

## 4. Giao thức Kaggle 20 epoch

Hai notebook chạy độc lập, cùng profile configs/kaggle_equal_20ep.json, seed 42, cùng split và tiêu chí chọn checkpoint: emotion27 Macro-F1 validation tại 0,5; hòa chọn epoch sớm nhất. Tất cả neural chạy đủ 20 epoch, không early stopping. 28 nhãn dùng selection_macro_f1 trên IDs 0–26; metric 28 nhãn được báo cáo riêng.

TF-IDF: max_features=40.000, unigram/bigram, min_df=2, sublinear_tf; 27 hoặc 28 Logistic Regression OvR, liblinear, C=1, max_iter=1000. TF-IDF không có epoch tương đương neural. BiGRU: train từ đầu, embedding/hidden 128, dropout 0,3, max_length=128, batch=64, AdamW lr=0,001. PhoBERT-base-v2: full fine-tuning, max_length=256, batch=16, AdamW lr=2e-5, weight_decay=0,01, revision 86cd7fd4c148980922ac11a2cf5e257f2ba639e1. Neural dùng BCEWithLogitsLoss, sigmoid độc lập và gradient clipping 1,0; không có scheduler/warmup/AMP được cấu hình.

Hai môi trường manifest cùng Tesla T4, Python 3.13.16, Torch 2.11.0+cu130, Transformers 4.57.3, scikit-learn 1.6.1. Code dùng một GPU mỗi process dù Kaggle hiển thị T4 x2. Một seed và hai session riêng chưa đo độ biến thiên.

## 5. Kết quả chính trên validation

### 5.1. Ba mô hình, 27 cảm xúc

{study27}

### 5.2. Ba mô hình, 28 nhãn gồm Neutral

{study28}

![Macro-F1 tại 0,5; mỗi panel giữ riêng tập nhãn và không dùng chênh lệch trực tiếp 27/28 như hiệu ứng Neutral.]({relative_fig}model_metrics.png)

PhoBERT cao nhất trong mỗi study theo bốn metric tổng hợp (Hamming Loss thấp hơn là tốt hơn). Macro-F1 27 của PhoBERT là {ph27.macro_f1:.6f}; Macro-F1 28 là {ph28.macro_f1:.6f}. Macro-F1 và Micro-F1 đánh giá dự đoán sau ngưỡng; LRAP đánh giá thứ hạng xác suất. Subset accuracy không dùng làm metric chính. Chưa có kiểm định ý nghĩa thống kê hoặc metric test.

### 5.3. Epoch, tham số và thời gian

{table(cost)}

Thời gian TF-IDF gồm fit vectorizer và LR, neural gồm train, validation từng epoch và ghi checkpoint; không gồm preprocessing/download. Không so tỷ lệ tốc độ CPU/GPU hoặc xem cùng epoch là cùng compute. PhoBERT 27 train khoảng {ph27.train_seconds/3600:.2f} giờ, 28 khoảng {ph28.train_seconds/3600:.2f} giờ. Chạy đủ 20 rồi nạp best: epoch chọn của PhoBERT là {int(ph27.best_epoch)} và {int(ph28.best_epoch)}; BiGRU chọn 17 và 20.

![Đường học từ bốn history 20 epoch: train BCE và tiêu chí chọn checkpoint chung emotion27 Macro-F1.]({relative_fig}learning_curves.png)

PhoBERT 27 có F1=0 ở epoch 1 nhưng về sau cải thiện, nên không kết luận lỗi từ epoch đầu. F1 validation biến động sau best trong khi train loss tiếp tục giảm. Không có validation loss; không tuyên bố hội tụ hoặc chẩn đoán overfitting chỉ từ đường loss train.

### 5.4. So sánh cùng 27 cảm xúc

{table(paired)}

Chiếu model 28 về cùng IDs 0–26, giữ cùng 2.066 ID/targets và ngưỡng 0,5. Kết quả: TF-IDF không đổi; BiGRU giảm {abs(paired.iloc[1].delta_pp):.4f} điểm phần trăm; PhoBERT giảm {abs(paired.iloc[2].delta_pp):.4f} điểm phần trăm. Với PhoBERT, Hamming Loss và LRAP trên 27 cảm xúc cũng kém hơn ở bản 28. Lượt đo này không cung cấp bằng chứng thêm Neutral cải thiện 27 cảm xúc. Cùng ngân sách và checkpoint criterion đã giảm nhiễu so sánh; vẫn chưa suy ra hiệu ứng nhân quả hay ý nghĩa thống kê từ một seed.

### 5.5. Neutral riêng

{table(neutral)}

PhoBERT có Neutral F1={ng.f1:.4f}, tìm được {int(ng.TP)}/94 mẫu, còn {int(ng.FN)} FN và {int(ng.FP)} FP. Neutral đã có đầu ra học được nhưng còn khó. Model 27 không có đầu ra Neutral: không tự gán Neutral F1=0 cho model đó.

## 6. Trả lời năm câu hỏi midterm bằng study 27 nhãn mới

### Q1. Nhãn nào có F1 cao/thấp nhất?

{table(pd.DataFrame(q1))}

Tất cả tại 0,5; xem phụ lục đủ từng nhãn. Các nhãn đồng hạng được liệt kê, không khái quát một nhãn dễ/khó cho mọi model.

### Q2. Tần suất train liên hệ với F1 ra sao?

{table(pd.DataFrame(q2))}

Spearman mô tả tương quan trên 27 nhãn; các nhãn phụ thuộc nhau. Không suy ra quan hệ nhân quả hoặc rằng chỉ tăng số mẫu sẽ giải quyết mọi nhãn khó.

### Q3. Model giữ được cặp thường đồng xuất hiện không?

{table(pd.DataFrame(pair_rows))}

Cặp disappointment–sadness chọn từ train trước khi đo validation. Joint recall là tỷ lệ mẫu có cả hai nhãn thật được đoán đủ cả hai. OvR không mô hình hóa cấu trúc nhãn trực tiếp; tỷ lệ thấp còn có thể liên quan ngưỡng bỏ sót.

### Q4. PhoBERT hơn BiGRU ở phong cách nào?

{table(styles)}

Trong các nhóm khảo sát, {top_style['style']} có chênh lệch Macro-F1 lớn nhất: {top_style.delta_pp:+.2f} điểm phần trăm. Nhóm short có ≤5 token khoảng trắng, emoji-heavy có ≥2 ký tự emoji theo regex và teencode theo từ điển heuristic; các nhóm chồng lấn và có nhiễu. Bảng dùng hai model 27 nhãn mới, không chuyển kết luận sang study khác.

### Q5. Nhạy với ngưỡng 0,5 đến đâu?

{table(pd.DataFrame(sensitivity))}

Khảo sát lại trên xác suất của study 27 mới, lưới 0,05–0,80 bước 0,05. Giữ checkpoint đã chọn tại 0,5; sweep chỉ là phân tích hậu nghiệm trên validation, không thay điểm bảng chính hay chọn lại checkpoint. LRAP không đổi khi đổi ngưỡng. Ngưỡng tối ưu trên validation chưa chứng minh cải thiện test, không áp thẳng cho model/tập nhãn khác.

## 7. Phân tích lỗi trên lượt mới

Đọc sáu ca PhoBERT 28: ba Neutral FP tự tin nhất và ba Neutral FN có xác suất thấp nhất. Chọn theo NPZ của lượt mới; không dùng lại các ca và kết luận dự đoán từ run cũ. Có hỗ trợ AI, nhóm cần xác nhận. Không xuất nguyên văn bình luận và không coi các ca chọn có chủ đích là mẫu ngẫu nhiên.

{table(review[['sample_id', 'case_type', 'observation', 'hypothesis']])}

Quan sát được tách với giả thuyết: ngữ cảnh hàm ý, câu ngắn, HTML và từ tục là hướng cần kiểm chứng bằng phân tích tiếp. CSV đi kèm có ground truth, dự đoán và xác suất Neutral. Mỗi model còn có 20 ứng viên FN và 20 FP tự sinh trong analysis; đó là danh sách để đọc, không phải bằng chứng tất cả đã được nhóm audit. Yêu cầu Final ít nhất 20 FN/20 FP vẫn chưa hoàn tất.

## 8. RQ và kế hoạch Final

Một ngưỡng chung 0,5 có tối ưu cho mọi nhãn không? Dùng neural backbone tốt nhất của midterm; hiện PhoBERT 27 có emotion27 Macro-F1 cao nhất. Khóa checkpoint, không retrain backbone khi đổi decoder; so sánh fixed 0,5, global tuned và per-label tuned. Chọn ngưỡng chỉ trên validation; khóa mọi lựa chọn rồi chấm held-out test đúng một lần. Phân tích theo tier/cardinality, profile ngưỡng, khoảng cách validation–test và ít nhất 20 FN/20 FP. Nếu tiếp tục dùng 28 nhãn, báo cáo riêng Neutral và phép chiếu 27.

## 9. Tái lập, nguồn và giới hạn

Hai ZIP Kaggle và notebook executed 08/09 là source of truth. scripts/import_equal20_results.py kiểm chứng manifest notebook khớp ZIP, sáu NPZ khớp ID/target/label order và metric CSV, history đủ 20, epoch best đúng tiêu chí và checkpoint có trong archive. Không nạp pickle/checkpoint, không train hay inference lại. Evidence được xuất bằng allowlist gồm JSON và NPZ ID-only.

Checksum file thô ở lượt 28 khác vì CRLF: đã xác minh từng checksum là biến thể LF/CRLF chính xác của source/config/CSV local. Không chỉ giả định hai file JSON có cùng nghĩa. Cấu hình sau parse và split IDs khớp giữa hai lượt; môi trường và GPU cùng thông tin manifest. provenance.json lưu checksum ZIP, notebook, thành viên đã trích và loại biến thể từng file; không sửa manifest gốc để ép khớp.

Giới hạn: một seed, checkpoint và khảo sát ngưỡng dùng validation; test chưa chấm; style heuristic; sáu ca lỗi có hỗ trợ AI cần nhóm xác nhận; chưa xác minh thuật toán split gốc và chưa có thông tin nhóm. Kết quả ít epoch được lưu như lịch sử, không là bằng chứng cho các bảng/kết luận chính hiện tại.

Lệnh tái tạo: python scripts/import_equal20_results.py; python scripts/build_report.py; python scripts/build_study_guide.py. Cần hai ZIP/notebook executed gốc và CSV được cấp quyền để import; các PDF/tổng hợp chỉ cần evidence đã kiểm chứng và EDA. Không phân phối raw text, vocabulary hoặc checkpoint công khai.

## Phụ lục A. F1 mọi nhãn của ba model

'''
    for labels in (27, 28):
        pivot = per[per.label_count == labels].pivot(index='label', columns='model', values='f1')
        pivot = pivot.reindex(LABELS if labels == 27 else LABELS+['neutral']).reset_index()
        text += f'### A.{labels}. {labels} nhãn, neural 20 epoch\n\n'+table(pivot[['label', *MODELS]])+'\n\n'
    text += '''## Tài liệu tham khảo

[1] Đề Project 2 ViGoEmotions và student-guide.pdf trong docs/, Midterm A–E và RQ1 Final.

[2] Hung et al., ViGoEmotions, EACL 2026, DOI 10.18653/v1/2026.eacl-long.129.

[3] Nguyen và Nguyen, PhoBERT, Findings of EMNLP 2020; VinAIResearch/PhoBERT, model vinai/phobert-base-v2.

[4] scikit-learn, PyTorch, Hugging Face Transformers và VnCoreNLP; phiên bản, revision và cấu hình theo manifest/run JSON trong evidence của study này.
'''
    (ROOT/'reports/midterm_report.md').write_text(text)
    register_fonts(); render_report(text)
    from build_equal20_presentation import build_presentation
    build_presentation()
    evidence_readme = '''# Evidence Kaggle 20 epoch — kết quả chính midterm

Hai ZIP kết quả và notebook 08/09 executed là nguồn. `provenance.json` ở thư mục cha lưu SHA-256, đối chiếu LF/CRLF, môi trường và hash các thành viên được trích. Thư mục này chỉ chứa JSON và NPZ với ID/nhãn/xác suất, không có raw text, vocabulary hoặc checkpoint.

Mọi model được chấm lại trên validation. Tham số/thời gian lấy nguyên từ run JSON; bốn history neural đủ 20 epoch và checkpoint selection dùng emotion27 Macro-F1 tại 0,5. Checkpoint được xác minh có trong ZIP riêng tư; không nạp lại hay train local.

`27/` và `28/` giữ manifest/config/status/comparison nguyên bản của từng ZIP và từng model. Bảng comparison gốc có đường dẫn Kaggle lịch sử; bảng chính được tính lại ở thư mục cha. Các phân tích trong `../analysis/28_emotion27/` chỉ chấm 27 cột cảm xúc của model 28, không phải metric đủ 28 nhãn. Test chưa chấm.
'''
    (OUT/'evidence/README.md').write_text(evidence_readme)
    (OUT/'README.md').write_text('''# Kết quả chính — Kaggle 20 epoch, 05/11/2026

- `model_comparison.csv`: sáu model, metric đủ tập nhãn, epoch, tham số, thời gian.
- `emotion27_comparison.csv`, `paired_emotion27_delta.csv`: đối chiếu cùng 27 cảm xúc.
- `per_label_metrics.csv`, `neutral_metrics.csv`: mọi nhãn và Neutral riêng.
- `label_extremes.csv`, `frequency_f1.csv`, `pair_preservation.csv`, `neural_style_comparison.csv`, `threshold_sensitivity.csv`: Q1–Q5 tính lại trên study 27 mới.
- `analysis/27_emotion27/` và `analysis/28_emotion27/`: evaluator chung trên 27 cột, đúng nguồn NPZ từng study; không ghi đè bảng lịch sử.
- `phobert28_error_review.csv`: sáu ca mới, AI hỗ trợ đọc; nhóm cần xác nhận.
- `evidence/`, `provenance.json`: dữ liệu dự đoán ID-only và kiểm chứng nguồn.
- Báo cáo chính ở `../midterm_report.md`; `figures/`, `slides.json` giữ hình và dữ liệu slide, không tạo thêm bản báo cáo trùng.

Build: `python scripts/import_equal20_results.py`, sau đó `python scripts/build_report.py`. Không train, không chấm test và không dùng metric các lượt ít epoch. Hai ZIP riêng tư gốc không được commit.
''')
    print('Built current equal20 midterm report and 15 English slides; no historical model scores used.')


if __name__ == '__main__':
    main()
