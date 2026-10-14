"""Build Vietnamese midterm report and slide PDFs from measured artifacts."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vigo-matplotlib')
import re
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfgen import canvas

from common import ROOT, LABELS

REPORTS = ROOT/'reports'
TABLES = REPORTS/'tables'


def table(df, columns=None):
    if columns:
        df = df[columns]
    def fmt(v):
        if pd.isna(v):
            return '—'
        return f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
    lines = ['| '+' | '.join(map(str,df.columns))+' |', '| '+' | '.join(['---']*len(df.columns))+' |']
    lines.extend('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
    return '\n'.join(lines)


def rich(text):
    text = escape(text)
    text = re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text = re.sub(r'`([^`]+)`',r'<font color="#185d78">\1</font>',text)
    return text


def register_fonts():
    fonts = Path(matplotlib.get_data_path())/'fonts/ttf'
    for name,file in [('Vigo','DejaVuSans.ttf'),('Vigo-Bold','DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('Vigo',normal='Vigo',bold='Vigo-Bold',italic='Vigo',boldItalic='Vigo-Bold')


def render_report(markdown):
    styles = getSampleStyleSheet()
    for key in ['Normal','Title','Heading1','Heading2','Heading3']:
        styles[key].fontName = 'Vigo-Bold' if key != 'Normal' else 'Vigo'
    styles['Normal'].fontSize=9;styles['Normal'].leading=14
    styles['Normal'].spaceAfter=7
    small = ParagraphStyle('TableVigo',parent=styles['Normal'],fontSize=7,leading=10,spaceAfter=0)
    story=[]; lines=markdown.splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',v) for v in cells):
                    rows.append([Paragraph(rich(v),small) for v in cells])
                i+=1
            t=Table(rows,colWidths=[495/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4f0f4')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#185d78')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')]),
                ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([t,Spacer(1,10)]);continue
        image_match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if image_match:
            caption,file=image_match.groups();p=REPORTS/file
            width,height=ImageReader(str(p)).getSize();scale=min(495/width,340/height)
            story.extend([Image(str(p),width*scale,height*scale),Paragraph(rich(caption),small),Spacer(1,10)])
        else:
            level=len(line)-len(line.lstrip('#'))
            key={1:'Title',2:'Heading1',3:'Heading2'}.get(level,'Normal')
            text=line[level:].strip() if level else line
            if text.startswith('- '):text='• '+text[2:]
            story.append(Paragraph(rich(text),styles[key]))
        i+=1
    def footer(c,doc):
        c.saveState()
        c.setFillColor(colors.white)
        c.rect(0,0,595,842,fill=1,stroke=0)
        c.setFont('Vigo',8);c.setFillColor(colors.HexColor('#526879'))
        c.drawString(50,28,'ViGoEmotions • ATI 2026 • Midterm | validation results')
        c.drawRightString(545,28,str(doc.page))
        c.restoreState()
    doc=SimpleDocTemplate(str(REPORTS/'midterm_report.pdf'),pagesize=(595,842),
                          leftMargin=50,rightMargin=50,topMargin=45,bottomMargin=48,
                          title='ViGoEmotions — Báo cáo giữa kỳ')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def render_slides(slides):
    c=canvas.Canvas(str(REPORTS/'midterm_slides.pdf'),pagesize=(960,540))
    c.setTitle('ViGoEmotions — Midterm presentation')
    md=['# ViGoEmotions — Nội dung thuyết trình giữa kỳ','']
    for idx,(title,bullets,figure,note) in enumerate(slides,1):
        c.setFillColor(colors.HexColor('#f6f8fa'));c.rect(0,0,960,540,fill=1,stroke=0)
        c.setFillColor(colors.HexColor('#153b50'));c.rect(0,440,960,100,fill=1,stroke=0)
        heading=Paragraph(escape(title),ParagraphStyle('slidehead',fontName='Vigo-Bold',fontSize=27,
                                                    leading=33,textColor=colors.white))
        heading.wrap(870,80);heading.drawOn(c,45,468)
        x,y,width=48,410,400 if figure else 860
        style=ParagraphStyle('slide',fontName='Vigo',fontSize=17,leading=25,textColor=colors.HexColor('#17394d'))
        for bullet in bullets:
            p=Paragraph('• '+rich(bullet),style);_,height=p.wrap(width,400)
            y-=height
            if y<55:raise ValueError(f'Slide {idx} text overflow')
            p.drawOn(c,x,y);y-=17
        if figure:
            p=REPORTS/figure;w,h=ImageReader(str(p)).getSize();scale=min(455/w,365/h)
            c.drawImage(str(p),475+(455-w*scale)/2,68+(365-h*scale)/2,
                        width=w*scale,height=h*scale,mask='auto')
        c.setFillColor(colors.HexColor('#607684'));c.setFont('Vigo',10)
        c.drawString(45,25,'ATI 2026  /  ViGoEmotions  /  Validation only')
        c.drawRightString(915,25,f'{idx:02d} / {len(slides):02d}')
        c.showPage()
        md += [f'## {idx}. {title}','',*['- '+b for b in bullets],'',f'Ghi chú thuyết trình: {note}','']
    c.save();(REPORTS/'midterm_slides.md').write_text('\n'.join(md),encoding='utf-8')


def main():
    summary=pd.read_csv(TABLES/'dataset_summary.csv')
    freq=pd.read_csv(TABLES/'label_frequency.csv')
    lengths=pd.read_csv(TABLES/'text_lengths.csv')
    metrics=[]
    for model,title in [('tfidf_lr','TF-IDF + OvR LR'),('gru','BiGRU'),('phobert','PhoBERT')]:
        path=TABLES/f'{model}_metrics.csv'
        if path.exists():
            row=pd.read_csv(path).query("split == 'val'").iloc[0].to_dict()
            row.update(model=title,status='Đã chạy')
        else:
            row=dict(model=title,status='Chưa chạy GPU',macro_f1=np.nan,micro_f1=np.nan,
                     hamming_loss=np.nan,lrap=np.nan,parameter_count=np.nan,train_seconds=np.nan)
        metrics.append(row)
    comparison=pd.DataFrame(metrics)[['model','status','macro_f1','micro_f1','hamming_loss','lrap','parameter_count','train_seconds']]
    comparison.to_csv(TABLES/'model_comparison.csv',index=False)
    neural_done=all((TABLES/f'{model}_metrics.csv').exists() for model in ('gru','phobert'))
    neural_styles=[]
    if neural_done:
        gru=pd.read_csv(TABLES/'gru_styles.csv').query("split == 'val'").set_index('style')
        pho=pd.read_csv(TABLES/'phobert_styles.csv').query("split == 'val'").set_index('style')
        for style in gru.index:
            neural_styles.append(dict(style=style,n=int(gru.loc[style,'n']),
                gru_macro_f1=gru.loc[style,'macro_f1'],phobert_macro_f1=pho.loc[style,'macro_f1'],
                delta_macro_f1=pho.loc[style,'macro_f1']-gru.loc[style,'macro_f1']))
    ns=pd.DataFrame(neural_styles,columns=['style','n','gru_macro_f1','phobert_macro_f1','delta_macro_f1'])
    ns.to_csv(TABLES/'neural_style_comparison.csv',index=False)
    lr=comparison.iloc[0];analysis=json.loads((REPORTS/'tfidf_lr_analysis.json').read_text())
    per=pd.read_csv(TABLES/'tfidf_lr_per_label.csv')
    pairs=pd.read_csv(TABLES/'tfidf_lr_pair_preservation_val.csv')
    sweep=pd.read_csv(TABLES/'tfidf_lr_thresholds_val.csv')
    best=sweep.sort_values(['macro_f1','threshold'],ascending=[False,False]).iloc[0]
    styles=pd.read_csv(TABLES/'tfidf_lr_styles.csv')
    cards=pd.read_csv(TABLES/'tfidf_lr_cardinality_val.csv')
    run=json.loads((REPORTS/'tfidf_lr_run.json').read_text())
    tiers=freq.groupby('tier').agg(labels=('label','count'),minimum=('train_count','min'),maximum=('train_count','max')).reindex(['head','mid','tail']).reset_index()
    # Integer parameter counts are easier to read than scientific notation.
    display_comparison=comparison.copy()
    display_comparison['parameter_count']=display_comparison.parameter_count.map(lambda x:'—' if pd.isna(x) else f'{int(x):,}')
    status='Đã có kết quả của cả ba mô hình; cần rà soát diễn giải và thông tin nhóm trước nộp.' if neural_done else 'Đã hoàn thành EDA và chạy TF-IDF. BiGRU/PhoBERT có mã và notebook Colab, chưa có kết quả huấn luyện. Gói này chưa đáp ứng đủ yêu cầu thực nghiệm ba mô hình để nộp bản cuối.'
    if neural_done:
        candidates=ns[ns['style']!='all'].sort_values('delta_macro_f1',ascending=False)
        top=candidates.iloc[0]
        conclusion=(f'Chênh lệch lớn nhất trong các nhóm khảo sát: {top["style"]}, '
                    f'delta Macro-F1 = {top.delta_macro_f1:.4f}. '
                    'Delta dương nghĩa là PhoBERT cao hơn GRU; nếu không có delta dương thì '
                    'không có bằng chứng PhoBERT vượt GRU trong các nhóm này. '
                    'Đây là chênh lệch mô tả, chưa kiểm định ý nghĩa thống kê.')
        q4=table(ns)+'\n\n'+conclusion
    else:
        q4='Chưa thể kết luận PhoBERT vượt GRU ở phong cách nào vì chưa có hai lượt huấn luyện. Notebook Colab tự xuất bảng chênh lệch Macro-F1 theo cùng mẫu validation. Không dùng điểm từ bài báo làm kết quả của nhóm.'
    neural_summary = ''
    if neural_done:
        gr = json.loads((REPORTS/'gru_run.json').read_text())
        ph = json.loads((REPORTS/'phobert_run.json').read_text())
        neural_summary = (f"BiGRU chạy {len(gr['history'])} epoch, checkpoint tốt nhất epoch {gr['best_epoch']}, "
            f"thời gian {gr['train_seconds']:.2f} giây. PhoBERT chạy {len(ph['history'])} epoch, "
            f"checkpoint tốt nhất epoch {ph['best_epoch']}, thời gian {ph['train_seconds']/60:.2f} phút. "
            f"Thiết bị neural: {ph['device']}. PhoBERT có Macro-F1 validation cao nhất trong ba baseline "
            "tại ngưỡng 0,5. Cả hai chọn epoch cuối, nên chưa khẳng định đã hội tụ. "
            "Kết quả Colab đã được đối chiếu checksum dữ liệu, ID, nhãn và tính lại metric từ xác suất; "
            "không huấn luyện lại khi nhập kết quả. ZIP không chứa checkpoint/vocab nên cần lưu riêng chúng.")
    text=f'''# ViGoEmotions — Báo cáo giữa kỳ

ATI Course Project 2026 • Nhận diện đa nhãn cảm xúc tiếng Việt

**Trạng thái:** {status}

## 1. Mục tiêu và phạm vi

Đầu vào là một bình luận tiếng Việt; đầu ra là tập con của 27 cảm xúc. Một bình luận có thể biểu đạt nhiều cảm xúc đồng thời. Vì vậy mô hình dùng 27 quyết định nhị phân độc lập, không dùng softmax để buộc chọn một lớp. Neutral được biểu diễn bằng vector cảm xúc bằng 0 theo quy ước project.

Midterm chiếm 30% điểm môn học và là checkpoint thuyết trình. Hai tài liệu yêu cầu mô tả dữ liệu, EDA, split/preprocessing hợp lệ, ba baseline, các chỉ số phù hợp, phân tích lỗi, câu hỏi nghiên cứu Final và mã tái lập. Báo cáo này bám mục Midterm A–E của đề; phần Final chỉ trình bày thiết kế, không nhận là đã thực hiện thí nghiệm Final.

**Câu hỏi thực nghiệm giữa kỳ:** Baseline dựa trên từ vựng nhận diện tốt/khó những nhãn nào, có giữ được các cặp cảm xúc đồng xuất hiện không, và ngưỡng 0,5 ảnh hưởng thế nào đến kết quả?

## 2. Dữ liệu, quyền truy cập và kiểm tra chất lượng

Bộ dữ liệu được cấp trong thư mục data gồm 20.664 bình luận: train 16.531, validation 2.066, test 2.067. Ba cột là id (định danh), text (bình luận), labels (danh sách ID). Dữ liệu nguồn thuộc phạm vi nghiên cứu/học thuật phi thương mại theo đề. Người dùng cần xin quyền và chấp nhận điều khoản tại https://huggingface.co/datasets/uitnlp/vigoemotions. Không đính kèm CSV nguồn, văn bản gốc hay vocabulary vào gói công khai.

Không có trường thiếu hoặc ID trùng. Một văn bản xuất hiện ở train ID 1698 và test ID red000094, cùng nhãn approval. Pipeline so sánh NFC + gom khoảng trắng, loại bản train và giữ nguyên bản test; train thực nghiệm còn 16.530 mẫu. CSV gốc không bị sửa. Việc dò trùng chỉ dùng text, không dùng nhãn test để huấn luyện.

Có 306 mẫu neutral + cảm xúc (251 train, 19 validation, 36 test), 725 mẫu chỉ neutral và 1 mẫu test lặp ID nhãn. Neutral riêng lẻ thành vector 0; với mẫu mixed-neutral, bỏ ID 27 và giữ các cảm xúc. Nhãn lặp được gộp. Đây là quyết định xử lý nhất quán, không khẳng định đã sửa đúng ý định người gán nhãn. Notebook ban đầu đếm sai mixed-neutral do dùng print trong biểu thức boolean; đã sửa và kiểm tra lại.

{table(summary)}

Thứ tự ID 0–26 được kế thừa từ notebook kiểm tra dữ liệu ban đầu; ID 27 là neutral. Dataset card đang yêu cầu quyền truy cập nên vẫn cần người sở hữu dữ liệu đối chiếu mapping này với metadata gốc được cấp quyền. Không sắp nhãn lại theo alphabet.

### 2.1. Danh mục đủ 27 nhãn và tần suất

{table(freq,['label_id','label','train_count','val_count','test_count','tier'])}

Phân tầng theo thứ hạng tần suất train: head là hạng 1–9, mid là 10–18, tail là 19–27; nếu bằng nhau thì ưu tiên ID nhỏ. Tier chỉ dùng train, cố định trước đánh giá test. Tail là khái niệm tương đối trong bộ dữ liệu này, không hàm ý chỉ có vài mẫu.

{table(tiers)}

Tỷ lệ nhãn phổ biến nhất/hiếm nhất là {freq.train_count.max()/freq.train_count.min():.2f} lần. Mức lệch giữa các lớp không quá cực đoan; thách thức khác là nhãn dương thưa: trung bình {summary.iloc[0].cardinality_mean/27*100:.2f}% trong 27 đầu ra là dương trên train. Không nên lặp lại nhận định “mất cân bằng nghiêm trọng” mà không phân biệt hai dạng này.

![Tần suất nhãn tính trên train sau loại trùng.](figures/label_frequency.png)

### 2.2. Cardinality và độ dài văn bản

Label cardinality là số cảm xúc dương sau khi bỏ neutral và gộp nhãn lặp. Trung bình train {summary.iloc[0].cardinality_mean:.3f}, validation {summary.iloc[1].cardinality_mean:.3f}, test {summary.iloc[2].cardinality_mean:.3f}; lớn nhất 5. Có nhóm cardinality 0 nên cần quy ước metric rõ ràng.

{table(lengths)}

Độ dài token trong EDA đo bằng tách khoảng trắng, không phải số từ sau tách từ tiếng Việt hay số BPE token của PhoBERT. Trung vị train là 11 đơn vị khoảng trắng, 47 ký tự; đuôi dài đến 214 đơn vị và 906 ký tự. GRU giới hạn 128 token; PhoBERT giới hạn 256 BPE token. Pipeline neural ghi tỷ lệ bị cắt riêng, không suy ra nó từ số token khoảng trắng.

![Phân bố cardinality, token khoảng trắng và ký tự theo split.](figures/distributions.png)

### 2.3. Đồng xuất hiện và tương quan

Ma trận co-occurrence dùng Y_train.T × Y_train; đường chéo là tần suất nhãn. Ma trận correlation dùng Pearson trên biến nhị phân (phi). Hai ma trận chỉ xây từ train. Đếm đồng xuất hiện lớn có thể do nhãn vốn phổ biến; vì vậy cần xem cả phi, không suy ra quan hệ nhân quả.

{table(pd.read_csv(TABLES/'train_top_pairs.csv').head(8))}

![Tương quan phi giữa 27 cảm xúc, chỉ tính train.](figures/train_correlation.png)

### 2.4. Văn phong tiếng Việt phi chuẩn

Kiểm tra các ca validation cho thấy viết tắt/teencode, từ vay mượn, thiếu dấu câu, dấu lặp và emoji có thể góp nghĩa cảm xúc. Emoji có thể bổ sung sắc thái vui vào một phát biểu hối tiếc; cách viết gần giống một cụm từ quen thuộc có thể làm mô hình từ vựng dự đoán sai. Câu rất ngắn và câu hỏi lịch sự thường cần ngữ cảnh. Đây là nhận xét từ mẫu đã đọc, không phải thống kê gán nhãn ngôn ngữ toàn tập.

Để so sánh định lượng: short có ≤5 token khoảng trắng; emoji-heavy có ≥2 ký tự trong các dải Unicode emoji định nghĩa ở common.py; teencode có ít nhất một token trong từ điển heuristic. Các nhóm có thể chồng lấn. Regex có thể bắt nhầm từ hợp lệ và bỏ sót nhiều dạng mới; số đếm emoji là ký tự mã hóa, không phải cụm grapheme hoàn chỉnh.

## 3. Split và pipeline tiền xử lý

Giữ ba split được cung cấp, tỷ lệ xấp xỉ 80/10/10; xuất ID cố định ở splits/ và SHA-256 của CSV trong run JSON. Tất cả mô hình dùng cùng train sau loại trùng và cùng validation. Chuẩn hóa NFC, gom khoảng trắng, giữ dấu tiếng Việt, emoji, phủ định và teencode ở bước chung; không học từ vựng/IDF trên validation hoặc test.

TF-IDF và GRU dùng lowercase. TF-IDF lấy word unigram/bigram, min_df=2, tối đa 40.000 features, sublinear TF; regex token một ký tự trở lên. Do regex word-token, emoji/dấu câu không trở thành feature TF-IDF: đây là hạn chế có chủ đích được công bố. GRU học vocab trên train, dùng PAD=0, UNK=1, tối đa 30.000 mục. PhoBERT tách từ bằng VnCoreNLP rồi BPE, không lowercase thủ công.

**Giới hạn stratification:** chưa có mã tạo split ban đầu nên không thể chứng nhận chúng được iterative multi-label stratified. Audit tỷ lệ từng nhãn theo split đã có trong label_frequency.csv; chênh lệch tuyệt đối lớn nhất so với train là {freq.val_drift_pp.abs().max():.2f} điểm phần trăm ở validation và {freq.test_drift_pp.abs().max():.2f} ở test. Đây là kiểm tra mô tả, không chứng minh stratification. Cần xác nhận với nguồn cấp dữ liệu/giảng viên rằng giữ official split có đáp ứng mục A hay không. Nếu phải tạo split mới: nhóm văn bản trùng trước, dùng iterative multi-label stratification 80/10/10 với seed 42, khóa ID, rồi chạy lại tất cả mô hình; không trộn kết quả hai giao thức.

Test chỉ được audit schema/phân bố biên/trùng lặp; chưa dùng để tính kết quả mô hình hay chọn ngưỡng. Midterm dùng validation cho so sánh thăm dò để bảo toàn test cho RQ Final. Validation đã dùng chọn checkpoint/ngưỡng nên điểm này không phải ước lượng độc lập của khả năng tổng quát hóa.

## 4. Mô hình và giao thức thực nghiệm

TF-IDF + OvR Logistic Regression: C=1, solver liblinear, max_iter=1000, seed=42, 27 bộ phân loại nhị phân. Có {run['features']:,} features thực tế, {int(lr.parameter_count):,} tham số hệ số + bias; số IDF và từ điển không tính vào số tham số. Nếu tối ưu không hội tụ, script báo lỗi thay vì âm thầm xuất kết quả.

BiGRU: embedding 128, hidden 128 mỗi hướng, nối trạng thái cuối hai chiều, dropout 0,3, linear head 27 logits; pack sequence theo độ dài thật để padding không trở thành tín hiệu. AdamW, lr=0,001, batch=64, 10 epoch, gradient clip=1. Chọn checkpoint có Macro-F1 validation cao nhất tại ngưỡng 0,5, hòa chọn epoch sớm hơn.

PhoBERT: vinai/phobert-base-v2, revision khóa trong configs/midterm.json; head 27 logits, AdamW lr=2e-5, weight_decay=0,01, batch=16, 4 epoch, max_length=256, cùng quy tắc chọn checkpoint. Tách từ VnCoreNLP trước tokenizer theo model card. Cả hai neural model dùng BCEWithLogitsLoss rồi sigmoid, không dùng cross-entropy một nhãn. Chưa chạy sweep kiến trúc hoặc weighted loss để giữ baseline đơn giản.

Seed 42 được đặt cho Python, NumPy, PyTorch và DataLoader; cuDNN deterministic và tắt benchmark. Không hứa bitwise-identical giữa loại GPU hoặc phiên bản thư viện khác nhau. Run JSON ghi hardware, package versions, config, checkpoint epoch, thời gian và checksum dữ liệu. Máy chạy TF-IDF: {run['environment']['platform']}; Python {run['environment']['python']}.

### 4.1. Chỉ số

- Macro-F1: trung bình F1 của đủ 27 nhãn, nhãn không có dự đoán đúng lấy 0 theo zero_division=0; nhấn mạnh các lớp ít gặp.
- Micro-F1: gộp TP/FP/FN trên tất cả nhãn rồi tính F1; lớp phổ biến đóng góp nhiều hơn.
- Per-label F1: báo cáo riêng từng cảm xúc, kèm precision, recall, support trong CSV.
- Hamming Loss: số quyết định nhãn sai chia N × 27; càng thấp càng tốt. Có thể thấp dù recall kém do nhiều nhãn âm.
- LRAP: đánh giá thứ hạng từ xác suất liên tục, không dùng dự đoán đã nhị phân. Với mẫu không có nhãn dương, sklearn trả 1; cần đọc cùng F1 và Hamming, không diễn giải là neutral đã dự đoán đúng.
- Số tham số và thời gian train: đếm trực tiếp và đo wall-clock. TF-IDF gồm fit vectorizer + fit LR; neural gồm training, validation mỗi epoch và lưu checkpoint, không gồm tải mô hình/tách từ. Khác phần cứng và phạm vi đo nên không kết luận tốc độ tương đối tuyệt đối.

Subset accuracy không là metric chính vì sai một nhãn làm toàn bộ mẫu bị tính sai. Hamming Loss cũng không dùng độc lập: dự đoán mọi nhãn âm có thể có loss thấp nhưng F1 bằng 0.

## 5. Kết quả thực nghiệm trên validation

Tất cả dòng baseline dưới đây dùng threshold=0,5. Dấu “—” nghĩa là chưa có thực nghiệm, không phải điểm 0.

{table(display_comparison)}

{neural_summary}

### 5.1. Per-label F1: TF-IDF tại 0,5

{table(per,['label','support','precision','recall','f1'])}

**Q1 — Nhãn tốt nhất/kém nhất:** gratitude có F1={per.set_index('label').loc['gratitude','f1']:.4f}; tiếp theo embarrassment và pride. Realization có F1=0, rồi disapproval và disappointment ở mức rất thấp. Disappointment khá phổ biến nhưng recall rất thấp, cho thấy chỉ có nhiều dữ liệu chưa đảm bảo mô hình tuyến tính nhận diện tốt.

**Q2 — Tần suất liên hệ với hiệu năng thế nào?** Spearman giữa số mẫu dương train và F1 validation trên 27 nhãn là rho={analysis['frequency_f1_spearman']:.3f}, p={analysis['spearman_pvalue']:.3f}. Quan hệ dương yếu; với thiết kế này chưa có bằng chứng thống kê mạnh về quan hệ đơn điệu. Đây là tương quan mô tả trên 27 nhãn phụ thuộc nhau, không phải chứng minh quan hệ nhân quả.

### 5.2. Phụ thuộc nhãn và cardinality

**Q3 — Baseline có giữ đồng xuất hiện?** Chọn 10 cặp phổ biến nhất chỉ từ train; sau đó đo cùng cặp trên validation. Joint recall là số mẫu dự đoán đúng cả hai nhãn / số mẫu thật có cả hai. Không xây ma trận test.

{table(pairs.head(6))}

Disappointment–sadness có 904 mẫu train; trên validation chỉ 2/122 ca thật được dự đoán đủ hai nhãn. OvR có thể tạo phụ thuộc gián tiếp qua đặc trưng chung nhưng không tối ưu cấu trúc nhãn; ngưỡng 0,5 đang bỏ sót nhiều cặp. Không kết luận mô hình “không học được bất kỳ phụ thuộc nào” chỉ từ thống kê này.

{table(cards,['cardinality','n','macro_f1','micro_f1','hamming_loss'])}

Nhóm ≥4 cảm xúc chỉ có 55 mẫu validation, nên kết quả biến động lớn. Nhóm cardinality=0 không có cảm xúc dương; F1=0 theo quy ước thư viện không đo trực tiếp chất lượng nhận diện neutral. Cần đọc Hamming cho nhóm này.

### 5.3. Phong cách văn bản

{table(styles,['style','n','macro_f1','micro_f1','hamming_loss','lrap'])}

TF-IDF có Macro-F1 thấp hơn trên emoji-heavy và teencode so với toàn validation. Đây là mô tả theo nhóm, chưa chứng minh các đặc trưng này là nguyên nhân vì phân bố nhãn/cỡ mẫu giữa các nhóm khác nhau. Tập short lại đạt điểm cao hơn tổng thể, nên không nên mặc định mọi câu ngắn đều khó.

**Q4 — PhoBERT vượt GRU rõ nhất ở nhóm nào?**

{q4}

### 5.4. Độ nhạy của ngưỡng

**Q5 — Threshold 0,5 có nhạy không?** Thử lưới 0,05–0,80, bước 0,05 trên cùng xác suất validation, không huấn luyện lại. Trong lưới này, Macro-F1 tốt nhất tại {best.threshold:.2f}: {best.macro_f1:.4f} thay vì {lr.macro_f1:.4f}, tăng {(best.macro_f1-lr.macro_f1)*100:.2f} điểm phần trăm. Micro-F1 là {best.micro_f1:.4f}; Hamming Loss tăng từ {lr.hamming_loss:.4f} lên {best.hamming_loss:.4f}. Hạ ngưỡng cải thiện F1 nhưng đánh đổi thêm quyết định nhãn sai. LRAP giữ nguyên vì thứ hạng xác suất không đổi.

{table(sweep[sweep.threshold.isin([.1,.15,.2,.35,.5])])}

![Khảo sát ngưỡng và quan hệ tần suất–F1, chỉ validation.](figures/tfidf_lr_diagnostics.png)

Không gọi 0,15 là ngưỡng tối ưu tổng quát: nó chỉ tốt nhất trên lưới và validation đã dùng để chọn. Cần khóa ngưỡng rồi kiểm tra trên test ở Final.

## 6. Phân tích lỗi và trường hợp biên

File tfidf_lr_error_cases_val.csv chứa 20 FN có xác suất thấp nhất và 20 FP có xác suất cao nhất, kèm ID, nhãn thật, nhãn dự đoán, xác suất, cardinality và style flags. Một mẫu có thể xuất hiện nhiều lần vì lỗi tính theo cặp mẫu–nhãn. Mẫu này ưu tiên lỗi tự tin, không đại diện ngẫu nhiên cho toàn bộ lỗi. Đã đọc thủ công 10 bình luận khác nhau trong danh sách; bảng dưới dùng mô tả trừu tượng, không chép nguyên bình luận.

| ID validation | Lỗi quan sát | Diễn giải thận trọng |
| --- | --- | --- |
| 3069 | FN pride; true cardinality 4 | Lời khen thân mật/slang; baseline bắt được một số cảm xúc tích cực nhưng thiếu pride/admiration. Có thể cần quan hệ giữa người nói và đối tượng. |
| 4712 | FN approval | Lời khen pha yêu cầu so sánh; mô hình không vượt ngưỡng ở nhãn nào. Chưa đủ bằng chứng để quy lỗi riêng cho ngưỡng. |
| 1930 | FN realization | Nhận xét mang ý suy ra nguyên nhân, có viết phi chuẩn; nét nghĩa suy luận không dễ biểu diễn bằng n-gram. |
| 2159 | FN caring | Câu hỏi ngắn kèm emoticon; ngữ cảnh thiếu nên caring khó tách khỏi sắc thái đùa. |
| tik011933 | FN amusement, giữ được remorse | Emoji bổ sung cảm xúc ngoài từ chính; word-token TF-IDF bỏ emoji, là giả thuyết nguyên nhân phù hợp. |
| 1148 | FP annoyance, giữ được anger | Đoạn dài với từ ngữ mạnh; annoyance gần anger/disgust về ngữ nghĩa nhưng không có trong nhãn chuẩn của mẫu. Không tự sửa ground truth. |
| tik014999 | FP remorse | Biểu đạt mong muốn/hối tiếc rất ngắn; nhầm các cảm xúc gần nhau khi thiếu ngữ cảnh. |
| tik008648 | FP gratitude, ground truth sadness | Viết phi chuẩn và nhiều emoji; khả năng mô hình dựa quá mạnh vào tín hiệu từ vựng gần nghĩa. |
| 1031 | FP love | Lời khen đối tượng được gán joy/admiration; mô hình dự đoán cảm xúc tích cực gần nhưng không khớp nhãn. |
| tik003388 | FP curiosity | Câu hỏi phục vụ đề nghị giúp đỡ; hình thức hỏi không nhất thiết biểu đạt tò mò. |

Các nhận định trên là giả thuyết từ đọc ca lỗi, chưa phải kết quả ablation. Có thể kiểm chứng ở Final bằng so sánh xử lý emoji, masking từ khóa hoặc phân tích attention chỉ như bằng chứng phụ; ưu tiên hoàn tất RQ bắt buộc trước.

## 7. Câu hỏi nghiên cứu cho Final và thiết kế kiểm chứng

**RQ1:** Một ngưỡng chung 0,5 có tối ưu cho đủ 27 cảm xúc khi phân bố nhãn khác nhau và nhãn dương thưa không?

Giả thuyết: ngưỡng global/per-label được chọn trên validation có thể tăng Macro-F1, nhưng per-label dễ overfit ở nhãn ít mẫu. Kết quả TF-IDF giữa kỳ chỉ là động lực khảo sát, không thay thế thí nghiệm trên neural backbone.

- Chọn kiến trúc neural có Macro-F1 validation cao nhất theo giao thức midterm và đóng băng checkpoint. Không chọn bằng test, không retrain backbone khi đổi decoder.
- Lưu xác suất validation. So sánh fixed 0,5; global tuned tối đa Macro-F1; per-label tuned tối đa F1 từng nhãn. Khóa lưới và quy tắc hòa trước khi chạy; mặc định hòa chọn ngưỡng cao hơn để giảm FP.
- Chốt toàn bộ ngưỡng trước khi chạy test. Thực hiện một lần inference trên test và chấm cả ba decoder từ cùng xác suất; không thử nhiều vòng trên test.
- Báo cáo Macro/Micro/per-label F1, Hamming, theo tier và cardinality 1/2/3/≥4; tách thêm nhóm neutral-only. Vẽ profile 27 ngưỡng và gap F1 validation − test để quan sát overfitting.
- Audit ít nhất 20 FN, 20 FP của neural model, mẫu nhiều cảm xúc và cặp nhãn dễ lẫn; không thay audit TF-IDF hiện tại cho yêu cầu Final.
- Lưu ID, true vector, probability vector và ba vector decoded cùng cardinality; chia sẻ theo giới hạn giấy phép, không kèm raw text. Nếu đi paper-track, chạy ≥3 seed và báo cáo độ lệch chuẩn.

## 8. Tái lập, giới hạn và đối chiếu yêu cầu

Chạy README theo thứ tự: tests → analyze_data.py → train_tfidf.py → build_report.py. Notebook 04 xử lý setup Colab, GRU/PhoBERT, xuất bảng/hình/probability và tạo lại báo cáo/slide. {'Đã có và kiểm chứng run JSON/xác suất của cả hai mô hình neural chạy trên Colab GPU.' if neural_done else 'Mã neural chưa có kết quả chạy GPU.'}

| Yêu cầu | Minh chứng / trạng thái |
| --- | --- |
| A: số mẫu, 27 nhãn, frequency, cardinality, length | CSV và hình trong reports/tables, reports/figures; đã chạy |
| A: co-occurrence, correlation, văn phong | Ma trận train-only và mô tả/heuristic; đã chạy |
| A: stratified split protocol | ID cố định + prevalence audit; nguồn thuật toán split gốc chưa được xác minh |
| B: TF-IDF, recurrent, PhoBERT | TF-IDF đã chạy; neural {'đã có file kết quả' if neural_done else 'đã chuẩn bị Colab, chưa chạy'} |
| C: đủ metric, per-label, tham số, thời gian | Có cho mô hình đã chạy; tự sinh từ evaluator chung khi neural hoàn tất |
| D: năm câu hỏi | Q1–Q3/Q5 có bằng chứng TF-IDF; Q4 {'có bảng chênh lệch neural' if neural_done else 'chưa có bằng chứng neural'} |
| E: slide, code, EDA, error analysis, RQ, README | Đã tạo; cần rà soát ca lỗi neural và điền thông tin nhóm khi nộp |

Giới hạn chính: một seed, validation đã dùng chọn cấu hình/ngưỡng, chưa đo tổng quát hóa test, heuristic style chưa được gán nhãn thủ công, mapping ID cần đối chiếu metadata gốc, nguồn split chưa xác minh stratification. Quy ước 27 output khác thiết lập 28 output có neutral riêng trong bài báo gốc; không so sánh trực tiếp số điểm hai giao thức. Các nhãn cùng mẫu phụ thuộc nhau nên F1 theo nhãn và kiểm định tương quan cần được diễn giải thận trọng.

## Tài liệu tham khảo

[1] Đề môn học: 02-vigoemotions-vietnamese-emotion.pdf, mục Midterm A–E (tr. 1–2), Reproducibility Rules (tr. 4).

[2] ATI Course Project 2026 — Scoring & Submission Guide: student-guide.pdf, tr. 1–2.

[3] Hung, T. Q., Nam, P. T., Luu, S. T., Nguyen, K. V. (2026). ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion Detection on Vietnamese Texts. EACL. https://aclanthology.org/2026.eacl-long.129/

[4] Nguyen, D. Q., Nguyen, A. T. (2020). PhoBERT: Pre-trained language models for Vietnamese. https://aclanthology.org/2020.findings-emnlp.92/ ; checkpoint/model card: https://huggingface.co/vinai/phobert-base-v2

[5] scikit-learn: OneVsRestClassifier và multilabel metrics. https://scikit-learn.org/stable/modules/multiclass.html ; https://scikit-learn.org/stable/modules/model_evaluation.html

[6] Hugging Face Transformers v4.57.3: model loading/classification. https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/auto ; VnCoreNLP: https://github.com/vncorenlp/VnCoreNLP

Mã và tài liệu có hỗ trợ AI trong phiên hoàn thiện; số liệu TF-IDF/EDA lấy từ chạy thực tế. Nhóm cần tự đọc, xác nhận diễn giải và tuân thủ quy định khai báo hỗ trợ AI của học phần.
'''
    (REPORTS/'midterm_report.md').write_text(text,encoding='utf-8')
    slides=[
        ('ViGoEmotions | Midterm', ['Nhận diện đa nhãn: một bình luận có thể mang nhiều cảm xúc.','27 cảm xúc; neutral là vector 0.', 'Bằng chứng hiện có: '+('ba mô hình trên validation.' if neural_done else 'EDA + TF-IDF trên validation; neural chờ Colab.')],None,'Mở đầu bằng bài toán, phạm vi và trạng thái thực nghiệm; không trình bày neural như đã chạy nếu chưa có số liệu.'),
        ('Dữ liệu & kiểm tra chất lượng',['20.664 mẫu; split gốc 16.531 / 2.066 / 2.067.','Loại 1 mẫu train trùng test → train 16.530.','306 mẫu mixed-neutral; 1 mẫu lặp nhãn.','Giữ dữ liệu nguồn riêng; chỉ xuất số liệu và ID.'],None,'Giải thích lỗi đếm mixed-neutral trong notebook cũ và quy ước 27 output.'),
        ('Phân bố 27 nhãn',['9 nhãn mỗi tier theo tần suất train.','Amusement cao nhất; relief thấp nhất.',f'Chênh lệch max/min: {freq.train_count.max()/freq.train_count.min():.2f} lần.','Nhãn dương thưa hơn vấn đề chênh lệch giữa lớp.'],'figures/label_frequency.png','Tier tương đối; không nói tail chỉ có rất ít mẫu. Bảng 27 nhãn đầy đủ có trong báo cáo.'),
        ('Cardinality & độ dài',[f'Train: {summary.iloc[0].cardinality_mean:.3f} cảm xúc/mẫu.','Cardinality tối đa 5; có neutral-only.','Median: 11 token khoảng trắng, 47 ký tự.','Whitespace token ≠ từ tiếng Việt ≠ BPE token.'],'figures/distributions.png','Nói rõ đơn vị token và vì sao phải đo tỷ lệ truncation riêng khi chạy neural.'),
        ('Các cảm xúc thường đi cùng nhau',['Disappointment–sadness: 904 mẫu train.','Anger–annoyance: 699.','Sadness–grief: 620.','Ma trận chỉ xây trên train; phi không hàm ý nhân quả.'],'figures/train_correlation.png','Đếm đồng xuất hiện và tương quan trả lời hai câu hỏi khác nhau.'),
        ('Split & preprocessing',['Giữ split đã cấp; fit vocab/IDF trên train.','NFC + gom khoảng trắng; giữ dấu và phủ định.','GRU: whitespace; PhoBERT: VnCoreNLP + BPE.','Có audit prevalence; chưa chứng minh split gốc được stratified.','Test giữ cho Final; midterm phân tích validation.'],None,'Phân biệt dữ liệu mô tả test với sử dụng test chọn mô hình. Nêu giới hạn nguồn thuật toán split một cách rõ ràng.'),
        ('Ba baseline theo đề',['TF-IDF (1–2 grams) + OvR Logistic Regression.','BiGRU: embedding 128, hidden 128 mỗi hướng, 10 epoch.','PhoBERT base-v2: revision cố định, 4 epoch.','Neural: 27 logits + BCE; sigmoid khi dự đoán.','Chọn checkpoint theo validation Macro-F1 ở 0,5.'],None,'Không dùng softmax. Nêu trạng thái Colab của hai neural model.'),
        ('Kết quả baseline | validation',[f'TF-IDF Macro-F1: {lr.macro_f1:.4f}; Micro-F1: {lr.micro_f1:.4f}.',f'Hamming Loss: {lr.hamming_loss:.4f}; LRAP: {lr.lrap:.4f}.',f'{int(lr.parameter_count):,} hệ số + bias; train {lr.train_seconds:.2f} giây CPU.',(f'BiGRU Macro-F1: {comparison.iloc[1].macro_f1:.4f}; PhoBERT: {comparison.iloc[2].macro_f1:.4f}.' if neural_done else 'GRU/PhoBERT chưa chạy.'),'Subset accuracy không là metric chính.'],None,'Thời gian TF-IDF gồm vectorizer + LR, không so tốc độ tuyệt đối với GPU. LRAP trên mẫu all-zero có quy ước đặc biệt.'),
        ('Nhãn nào tốt, nhãn nào khó?',['Gratitude F1 0,6797; embarrassment 0,6311; pride 0,5500.','Realization F1 0; disapproval 0,0160; disappointment 0,0280.',f'Tần suất–F1: Spearman rho {analysis["frequency_f1_spearman"]:.3f}, p {analysis["spearman_pvalue"]:.3f}.','Tần suất cao chưa đủ; disappointment là ví dụ.'],None,'Các số ở đây chỉ nói về TF-IDF 0,5. Không khái quát các kết luận riêng của TF-IDF sang PhoBERT.'),
        ('Ngưỡng 0,5 ảnh hưởng mạnh',[f'Khảo sát 0,05–0,80 trên validation.',f'Ngưỡng {best.threshold:.2f}: Macro-F1 {best.macro_f1:.4f}.',f'Tăng {(best.macro_f1-lr.macro_f1)*100:.2f} điểm phần trăm.',f'Hamming tăng lên {best.hamming_loss:.4f}.','Chưa chứng minh cải thiện trên test.'],'figures/tfidf_lr_diagnostics.png','LRAP không đổi khi đổi threshold. Tăng F1 không đồng nghĩa mọi metric cùng tốt hơn.'),
        ('Baseline giữ được cặp nhãn?',['Disappointment–sadness: chỉ 2/122 ca validation đúng đủ cặp.','Joint recall = 1,64% tại ngưỡng 0,5.','OvR không có cơ chế học cấu trúc nhãn trực tiếp.','Cần phân biệt phụ thuộc nhãn với vấn đề ngưỡng bỏ sót.'],None,'Cặp được chọn từ train trước, rồi đo trên validation; không xây ma trận test.'),
        ('Phong cách & lỗi biên',[f'TF-IDF Macro-F1: short 0,3310; emoji-heavy 0,1690; teencode {styles.set_index("style").loc["teencode","macro_f1"]:.4f}.','Emoji bị bỏ bởi word-token TF-IDF; giả thuyết cho FN amusement.','Câu hỏi giúp đỡ có thể bị FP curiosity.','Có 20 FN + 20 FP; đã đọc thủ công 10 ca khác nhau.','Q4 PhoBERT vs GRU: '+(f'teencode có delta cao nhất: {top.delta_macro_f1*100:.2f} điểm phần trăm.' if neural_done else 'chờ kết quả Colab.')],None,'Style groups chồng lấn và có heuristic nhiễu. Không chép comment thô lên slide; mã ca lỗi trong báo cáo giúp kiểm tra nội bộ.'),
        ('RQ Final: một ngưỡng có đủ?',['Chọn neural backbone tốt nhất từ midterm; giữ nguyên trọng số.','So sánh fixed 0,5 / global tuned / per-label tuned.','Tune validation → khóa ngưỡng → test đúng một lần.','Phân tích theo tier/cardinality, profile ngưỡng và val–test gap.'],None,'Bằng chứng TF-IDF tạo động lực, nhưng đề yêu cầu Final dùng neural architecture tốt nhất.'),
        ('Việc cần chốt trước khi nộp',[('Đã kiểm chứng kết quả Colab; lưu riêng checkpoint và vocab.' if neural_done else 'Chạy notebook 04 trên Colab GPU.'),'Đối chiếu ID-to-label với metadata nguồn; xác nhận split protocol.','Đọc error analysis neural; bổ sung tên/MSSV/phân công nhóm.','Gói nộp: PDF, slide, code, README, config, ID, bảng/hình, probabilities.'],None,'Nêu trạng thái thật và công việc cụ thể còn lại. Nếu đã chạy Colab, chuyển trọng tâm sang rà soát nội dung.'),
        ('Nguồn & tái lập',['Đề ViGoEmotions và student-guide: tiêu chí đánh giá.','Hung et al., EACL 2026: doi.org/10.18653/v1/2026.eacl-long.129.','PhoBERT: Nguyen & Nguyen, 2020; vinai/phobert-base-v2.','scikit-learn, Transformers, VnCoreNLP: dẫn nguồn trong báo cáo.','Seed 42; package versions, checksum, hardware lưu trong run JSON.'],None,'Các thư viện và pretrained checkpoint phải được trích dẫn. Điểm paper không phải điểm chạy của nhóm.')]
    register_fonts();render_report(text);render_slides(slides)
    print('Wrote midterm_report.md/pdf, midterm_slides.md/pdf, model comparison tables')


if __name__ == '__main__':
    main()
