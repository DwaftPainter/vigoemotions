# AI-assisted qualitative audit — 62 ca test PhoBERT 28

Ngày đọc: **17/11/2026**. Người thực hiện: **Codex, AI assistant**. Đã đọc nội dung private của **62 bình luận khác nhau**, đối chiếu annotation và ba decoder cố định, ghi nhận xét riêng cho từng ca. Đây là AI-assisted review; không phải 62 xác nhận của thành viên nhóm.

Nguồn: [cases](cases.csv), [nhận xét đủ 62 ca](ai_assisted_review.csv), [summary](ai_assisted_summary.json), [provenance](ai_assisted_provenance.json). File có cả source text và nhận xét AI nằm private ở `data/final28/human_audit/ai_review_private.csv`. Worksheet dành cho người và trạng thái 0 human reviews được giữ nguyên.

## 1. Phạm vi và cách đọc

- 20 core FN và 20 core FP của per-label decoder, giữ nguyên mẫu hash đã chuẩn bị.
- 2 Neutral errors, 5 ca true cardinality ≥4 và 15 ca cho ba cặp cảm xúc chọn từ retained train.
- Các nhận xét dưới đây diễn đạt trừu tượng; không trích bình luận gốc hoặc công khai tên được nhắc đến trong dữ liệu.
- `reference_assessment` đánh giá target label ở các ca FN/FP, toàn bộ ca ở các dòng context. `questionable` nghĩa là cần xem lại ranh giới/độ đầy đủ của reference, không khẳng định annotation sai. `insufficient_context` nghĩa là không đủ căn cứ từ comment riêng lẻ.
- Metrics vẫn dùng reference gốc. Không sửa nhãn, scores, thresholds hoặc lựa chọn decoder từ những nhận xét này.

## 2. Lỗi đã có ở fixed hay mới do per-label?

Đây là đối chiếu quyết định chính xác từ saved outputs, khác với phần diễn giải ngôn ngữ của AI.

| Mẫu chính | Lỗi vẫn có ở fixed | Lỗi mới so với fixed | Tổng |
| --- | ---: | ---: | ---: |
| Core FN | 20 | 0 | 20 |
| Core FP | 15 | 5 | 20 |

Năm FP mới là annoyance ở sample `171`, `12539`, `you003220`; nervousness ở `1011`; love ở `thr000159`. Các target scores đều dưới 0,5 nhưng đạt cutoff per-label. Trong 20 FN, global phục hồi target ở `3777` và `13789`, còn per-label vẫn bỏ sót.

Ví dụ `you003220`: score annoyance **0,051592**, cutoff per-label **0,05**. Fixed/global không dự đoán annoyance, per-label thêm nhãn này. Đọc nội dung cho thấy giọng kể hài và không có sự bực bội tường minh; đây là một FP mới có lý do semantic để xem xét. Điều đó chưa chứng minh cue nào làm model cho điểm.

Ví dụ `1011`: nervousness được thêm ở cutoff **0,1**, nhưng văn bản đã có fear và hành vi tránh né. FP theo annotation là sự kiện xác định; việc nervousness thực sự không phù hợp về nghĩa vẫn cần người xác nhận. Hai khái niệm này cần được tách khi bảo vệ kết quả.

Trong mẫu 40 cặp chính, phần lớn lỗi đã tồn tại ở fixed; không thể giải thích tất cả bằng việc per-label đổi threshold. Các con số trên chỉ mô tả mẫu audit, không phải ước lượng tỷ lệ lỗi mới của toàn bộ test.

## 3. Những pattern đọc được

| Nhóm nhận xét AI | Số ca |
| --- | ---: |
| Giao thoa/rìa nghĩa giữa các nhãn (`label_overlap`) | 24 |
| Slang/emoji (`slang_emoji`) | 8 |
| Cảm xúc hàm ý (`implicit_emotion`) | 7 |
| Mỉa mai/tự trào (`sarcasm`) | 7 |
| Thiếu ngữ cảnh (`missing_context`) | 6 |
| Đánh đổi ở quyết định threshold (`threshold_tradeoff`) | 4 |
| Phủ định/điều kiện/thái độ với lời được nhắc (`negation`) | 3 |
| Ca đối chiếu hoặc nhóm khác (`other`) | 3 |

Mỗi ca được gán một nhóm chính để tổ chức nhận xét; nhiều hiện tượng có thể cùng xuất hiện. Đây không phải taxonomy nguyên nhân được kiểm chứng, cũng không phải tần suất population.

### 3.1. Giao thoa nhãn và cảm xúc phụ

Ở `4526`, lời tán thành được model chuyển thành pride thay vì approval. Ở `13124`, dấu hiệu suy sụp được giữ như remorse dù không có lời ăn năn rõ. Ở `32504`, lời nói về phẩm chất được model giữ như caring; cần phân biệt nói về một phẩm chất và bộc lộ cảm xúc quan tâm.

Các ca này phù hợp giả thuyết về ranh giới nhãn, chưa chứng minh cơ chế representation. Có ca chiều ngược lại: `you001037` có lời ủng hộ và ký hiệu trái tim, khiến love là một cách hiểu có cơ sở dù không có trong reference. `x000137` có lời động viên trực tiếp, nên caring ngoài reference cũng không thể bị coi là sai nghĩa chắc chắn.

### 3.2. Cảm xúc hàm ý và giọng nói

`28445` có lời chúc trong ngữ cảnh buồn; model giữ caring và các nhãn tích cực nhưng bỏ sadness. `28083` dùng giọng mỉa mai; đầu ra nghiêng về amusement/realization thay vì disappointment. `tik023375` là một so sánh phóng đại; model chỉ giữ disappointment thay vì amusement và realization của reference.

Ở `28909`, bề mặt cười đi cùng lời chế giễu làm anger khó xác nhận từ câu đơn lẻ. Đây là lý do cần tách một giả thuyết về sarcasm khỏi khẳng định model đã hiểu sai cảm xúc thật.

### 3.3. Phủ định, điều kiện và thái độ với nội dung được nói

`tik017852` bàn về sự tự tin giả vờ và những lời tự khích lệ; model dự đoán pride rất cao. `6680` phân biệt cảm xúc trước đây với việc không còn cảm giác, nhưng per-label thêm love. `451` nhắc giận trong một câu hỏi phủ định/tự trào; không thể coi mọi lần xuất hiện từ chỉ giận là bằng chứng anger đang được trải nghiệm.

Các ca phù hợp hướng kiểm tra phạm vi phủ định và đối tượng/thời điểm cảm xúc. Chưa thực hiện can thiệp văn bản hoặc ablation để xác định nguyên nhân model cho điểm.

### 3.4. Emoji và comment quá ngắn

`tik021172` chỉ có một emoji: khó xác nhận excitement, love hoặc admiration mà không có tình huống. `28483` dùng emoji và lời cầu xin; model giữ gratitude nhưng bỏ desire. `48` có ký hiệu cười nhưng model vẫn giữ sadness và grief; vì vậy không có quy luật mọi emoji cười đều kéo model sang amusement.

`15225` rất ngắn nhưng có bốn reference labels. Joy được giữ, còn relief cần thông tin về trạng thái trước đó để đánh giá. Không thể suy rằng mọi FN ở ca cardinality cao là semantic failure rõ ràng chỉ từ reference.

## 4. Annotation plausibility: đánh giá AI có giới hạn

| Nhận định AI | Core FN | Core FP | Toàn bộ 62 ca |
| --- | ---: | ---: | ---: |
| Có cơ sở phù hợp reference (`consistent`) | 11 | 9 | 32 |
| Cần xem lại ranh giới/độ đầy đủ (`questionable`) | 4 | 8 | 16 |
| Chưa đủ ngữ cảnh (`insufficient_context`) | 5 | 3 | 14 |

Các cột không có cùng đơn vị đánh giá: core rows tập trung target, context rows xem cả ca. Không diễn giải 16/62 thành annotation error rate hoặc 32/62 thành tỷ lệ annotation đúng.

`you001985` là ví dụ cần xác nhận: người viết hỏi lời khuyên trong một tình huống có cảm xúc tiêu cực, reference chỉ có confusion, model giữ sadness với score cao. Sadness có thể có cơ sở semantic; điều đó không cho phép sửa nhãn bằng ý kiến AI rồi chấm lại test.

## 5. Cardinality cao, cặp cảm xúc và Neutral

### Cardinality cao

Cả năm ca high-cardinality đã chọn còn ít nhất một FN ở per-label. `18169` giữ được bốn cảm xúc nhưng bỏ mong muốn; `12474` phục hồi love so với fixed nhưng vẫn thiếu nervousness và thêm grief; `6293` giữ love/desire/fear nhưng thiếu nervousness và thêm nhãn ngoài reference.

Những ví dụ này cho thấy một output có nhiều nhãn đúng vẫn có thể bỏ một trạng thái phụ. Riêng `15225` và `32486` cần ngữ cảnh để xác nhận đầy đủ reference. Không dùng năm ca này để ước lượng tỷ lệ bỏ sót của cả nhóm ≥4.

### Cặp cảm xúc

Điều kiện chọn ca là có đủ hai **true labels hoặc per-label predicted labels**. Do đó không được coi năm ca mỗi cặp là năm positive ground-truth pairs.

| Cặp chọn từ train | Đủ true và đủ per-label pred | Chỉ đủ true | Chỉ đủ per-label pred |
| --- | ---: | ---: | ---: |
| Disappointment–sadness | 4 | 0 | 1 |
| Anger–annoyance | 0 | 3 | 2 |
| Sadness–grief | 3 | 1 | 1 |

`3351` và `743` là ca đối chiếu: cả ba decoder giữ đủ bộ nhãn reference, nên audit không chỉ chứa failure examples. Với anger–annoyance, ba ca có đủ true pair đều thiếu ít nhất một nhãn trong per-label; hai ca còn lại là predicted pair với một nhãn ngoài reference. Đây là mô tả mẫu được chọn, không phải joint recall trên toàn test và không chứng minh đã/không học phụ thuộc nhãn.

### Neutral

Ở `247`, realization được giữ nhưng Neutral bị bỏ; mixed annotation được giữ nguyên. Ở `11588`, một yêu cầu ngắn có hàm phê bình chỉ được model giữ như Neutral; mức disapproval cần ngữ cảnh để xác nhận. Không ép Neutral loại trừ emotion hoặc tự gán Neutral cho empty output. Audit có ba ca mixed Neutral và không xem mixed labels tự chúng là lỗi.

## 6. Câu viết có thể dùng cho báo cáo

“AI-assisted qualitative review of 62 selected comments suggested recurring ambiguity between related emotion labels, implicit or ironic affect, and the scope of negation or conditional statements. In the 40 primary audited label-instance pairs, all 20 false negatives and 15 of the 20 false positives were already errors under fixed thresholding; five false positives were introduced by per-label thresholding. These case-level observations do not establish causal mechanisms or population error rates. Several apparent errors were semantically plausible predictions absent from the reference labels, motivating human adjudication rather than post-test relabeling.”

Đoạn trên phải giữ nhãn **AI-assisted** và giới hạn của mẫu khi đưa vào paper. Trạng thái human audit vẫn là **0/62 xác nhận**. Người đọc có thể dùng notes AI để thảo luận, nhưng chỉ ghi reviewer/status hoàn tất sau khi thực sự đọc; không sao chép toàn bộ nhận xét rồi xem đó là xác nhận độc lập.
