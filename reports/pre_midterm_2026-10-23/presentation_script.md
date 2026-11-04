# ViGoEmotions — Kịch bản trình bày pre-midterm

Ngày trình bày: **23/10/2026**. Thời lượng mục tiêu: **8–10 phút**, chưa tính hỏi đáp. Lời thoại dùng “nhóm em”; các thành viên có thể chia nhau theo phần, không cần đọc nguyên văn.

Số liệu: lượt notebook 04 mới trong `vigoemotions_results.zip`, đã được kiểm tra từ xác suất validation, split ID, cấu hình, lịch sử epoch và logs. Chưa có kết quả notebook 05 mới được nhập và kiểm chứng. Nếu Colab đã chạy xong 05, chỉ bổ sung kết luận sau khi kiểm tra artifacts.

**Trước khi trình bày:** slide PDF cũ mang snapshot trước khi lượt 04 mới hoàn tất. Khi dùng lại, cần cập nhật slide 1, 7, 8, 10, 11: 04 đã hoàn tất; PhoBERT Macro-F1 **0,4679**; chênh lệch với BiGRU **6,18 điểm phần trăm**; chênh lệch teencode **10,91 điểm phần trăm**. Không đọc các số 0,4393 / 3,33 / 9,41 như kết quả lượt mới.

Chỉ phần **Lời nói** là nội dung trình bày. Phần **Thao tác** là hướng dẫn cho người điều khiển máy. Các đường dẫn minh chứng bên dưới mở từ thư mục chứa file này.

## Slide 1 — Giới thiệu đề tài · 25 giây

**Lời nói:**

“Em chào thầy/cô. Hôm nay nhóm em báo cáo tiến độ đề tài ViGoEmotions, nhận diện cảm xúc trong bình luận tiếng Việt. Đây là bài toán đa nhãn: một bình luận có thể đồng thời thể hiện nhiều cảm xúc.

Trong buổi này, nhóm em trình bày phần dữ liệu đã kiểm tra, các baseline đã chạy, kết quả ban đầu và công việc còn lại trước Midterm.”

**Thao tác:** Mở tên đề tài và cấu trúc notebook 01 đến 05; chưa cần chạy code trực tiếp.

## Slide 2 — Nhóm đã làm được gì? · 30 giây

**Lời nói:**

“Đến hiện tại, nhóm em đã hoàn thành bốn phần: kiểm tra dữ liệu và chuẩn hóa nhãn; khám phá dữ liệu; xây dựng và chạy ba baseline; lưu kết quả cùng metadata để đối chiếu.

Lượt chạy mới của notebook 04 đã hoàn tất và được kiểm tra. Phần chưa có kết luận mới là thử nghiệm learning rate ở notebook 05. Ngoài ra, nhóm còn cần bổ sung phân tích lỗi thủ công cho các mô hình neural.”

**Thao tác:** Chỉ vào ba dòng trong [bảng baseline mới](evidence/model_comparison.csv). Nói rõ đây là validation, không phải test.

## Slide 3 — Dữ liệu và kiểm tra chất lượng · 55 giây

**Lời nói:**

“Dữ liệu nguồn có 20.664 bình luận, gồm 16.531 mẫu train, 2.066 validation và 2.067 test. Nhóm phát hiện một văn bản trùng giữa train và test, nên loại bản train; số mẫu train thực nghiệm còn 16.530.

Mapping ID đã được đối chiếu với bảng nhãn trên Hugging Face: ID 0 đến 26 là cảm xúc, ID 27 là neutral. Theo quy ước của đề, nhóm dùng 27 đầu ra: neutral-only được mã hóa bằng vector toàn 0; nếu neutral xuất hiện cùng cảm xúc thì giữ cảm xúc và bỏ neutral.

Nhóm giữ split được cấp. Paper mục 5.1 chỉ công bố tỷ lệ 8:1:1, không mô tả thuật toán chia tập; vì vậy nhóm chưa khẳng định split đã được stratified.”

**Thao tác:** Mở [bảng mapping và đối chiếu paper](../../docs/label_mapping.md), chỉ mục nguồn và phần split; sau đó chỉ số mẫu trong [dataset summary](evidence/dataset_summary.csv). Paper ở trang PDF 6, trang in 2810.

## Slide 4 — EDA cho thấy điều gì? · 40 giây

**Lời nói:**

“Trên train, một mẫu có trung bình khoảng 1,859 nhãn cảm xúc. Điều này cho thấy không thể coi các nhãn là loại trừ nhau, nên nhóm dùng đầu ra sigmoid độc lập cho từng nhãn.

Tần suất các cảm xúc không đều. Vì vậy nhóm theo dõi Macro-F1 và F1 từng nhãn, thay vì chỉ nhìn một chỉ số tổng thể. Nhóm cũng khảo sát độ dài và chia các nhóm văn phong như bình luận ngắn, nhiều emoji và teencode để xem mô hình gặp khó ở đâu.”

**Thao tác:** Mở [hình tần suất nhãn](../figures/label_frequency.png) và chỉ các nhóm head/mid/tail. Các nhóm style là heuristic, chưa phải nhãn được con người xác minh riêng.

## Slide 5 — Quan hệ giữa các nhãn · 35 giây

**Lời nói:**

“Nhóm đã tính ma trận đồng xuất hiện và tương quan nhãn, chỉ bằng dữ liệu train. Ví dụ, disappointment và sadness cùng xuất hiện trong 904 mẫu train, với hệ số phi khoảng 0,362.

Từ đó, nhóm đặt câu hỏi: khi hai cảm xúc thường đi cùng nhau, mô hình có dự đoán đủ cả hai không? Đây là cơ sở để phân tích lỗi theo cặp nhãn, chứ không chỉ đánh giá từng nhãn riêng lẻ.”

**Thao tác:** Chỉ hàng đầu trong [bảng cặp nhãn](evidence/train_top_pairs.csv). Không diễn giải đồng xuất hiện thành quan hệ nhân quả.

## Slide 6 — Ba baseline và cách fine-tune · 65 giây

**Lời nói:**

“Nhóm triển khai ba mức baseline. Đầu tiên là TF-IDF kết hợp One-vs-Rest Logistic Regression, để có một mốc so sánh dựa trên đặc trưng từ. Thứ hai là BiGRU, học embedding và biểu diễn chuỗi từ đầu. Thứ ba là PhoBERT pretrained, được fine-tune toàn bộ backbone và lớp phân loại mới.

BiGRU được đặt ngân sách 10 epoch; PhoBERT là 4 epoch. Đây là cấu hình ban đầu, chưa được chứng minh tối ưu. Với neural model, nhóm dùng BCE, AdamW và đánh giá validation sau mỗi epoch. Checkpoint có Macro-F1 cao nhất tại ngưỡng 0,5 được giữ lại.

Trong lượt mới, checkpoint tốt nhất của BiGRU là epoch 10, PhoBERT là epoch 4. Việc cả hai nằm ở epoch cuối chưa chứng minh mô hình đã hội tụ; nhóm cần đọc đường học trước khi quyết định tăng epoch.”

**Thao tác:** Mở [config](../../configs/midterm.json), rồi [PhoBERT run JSON](evidence/phobert_run.json), chỉ `config`, `history` và `best_epoch`. Không đọc hết các tham số trên màn hình.

## Slide 7 — Kết quả validation mới · 55 giây

**Lời nói:**

“Đây là kết quả lượt chạy mới, trên cùng validation gồm 2.066 mẫu và cùng threshold 0,5.

Macro-F1 của TF-IDF là 0,2388; BiGRU là 0,4060; PhoBERT là 0,4679. PhoBERT cao hơn BiGRU khoảng 6,18 điểm phần trăm Macro-F1. Micro-F1 của PhoBERT là 0,5312; Hamming Loss là 0,0523 và LRAP là 0,7200.

Như vậy, PhoBERT tốt nhất trong ba baseline của lượt chạy này theo Macro-F1 tại ngưỡng 0,5. Đây vẫn là kết quả validation trên một seed, chưa phải kết quả test hoặc bằng chứng về mức chênh lệch có ý nghĩa thống kê.”

**Thao tác:** Mở [bảng baseline mới](evidence/model_comparison.csv), chỉ Macro-F1 trước, sau đó các chỉ số bổ sung. Không mở nhầm `reports/tables/model_comparison.csv` của bộ cũ.

## Slide 8 — Insight về văn phong và threshold · 55 giây

**Lời nói:**

“Khi phân tích văn phong trong lượt mới, PhoBERT hơn BiGRU nhiều nhất ở nhóm teencode trong ba nhóm được khảo sát. Macro-F1 lần lượt khoảng 0,4261 và 0,3170, chênh lệch 10,91 điểm phần trăm trên 307 mẫu.

Nhóm cũng thấy threshold ảnh hưởng mạnh đến kết quả. Với TF-IDF, ngưỡng 0,15 cho Macro-F1 validation khoảng 0,4844, cao hơn ngưỡng 0,5; nhưng Hamming Loss tăng. Vì thế không thể coi 0,5 là ngưỡng tối ưu cho mọi nhãn.

Điểm TF-IDF sau chọn ngưỡng không phải một so sánh cùng điều kiện với PhoBERT tại ngưỡng cố định. Đây là khảo sát để đặt câu hỏi nghiên cứu cho Final.”

**Thao tác:** Chỉ hàng `teencode` trong [bảng style mới](evidence/neural_style_comparison.csv), rồi các hàng threshold 0,15 và 0,5 trong [threshold sweep TF-IDF](evidence/tfidf_lr_thresholds_val.csv).

## Slide 9 — Phân tích lỗi đã đến đâu? · 40 giây

**Lời nói:**

“Ở TF-IDF tại threshold 0,5, mô hình chỉ dự đoán đúng đồng thời disappointment và sadness trong 2 trên 122 mẫu validation có cặp này. Đây là một biểu hiện bỏ sót khi một mẫu có nhiều cảm xúc.

Nhóm đã xuất danh sách ứng viên false negative và false positive. Phần audit thủ công hiện có là 10 bình luận TF-IDF khác nhau của lượt baseline trước. Nhóm chưa hoàn tất đọc lỗi neural của lượt mới, nên chưa coi các danh sách tự sinh là phân tích lỗi đã hoàn thành.”

**Thao tác:** Chỉ `val_true_count` và `joint_true_positive` trong [bảng bảo toàn cặp TF-IDF](evidence/tfidf_lr_pair_preservation_val.csv). Không chiếu bình luận thô.

## Slide 10 — Minh chứng và khả năng tái lập · 35 giây

**Lời nói:**

“Mỗi lượt chạy đều lưu cấu hình, seed, môi trường, checksum dữ liệu, lịch sử epoch và xác suất validation trước khi áp dụng threshold. Nhờ đó nhóm có thể tính lại metric từ xác suất để kiểm tra kết quả.

Notebook 04 mới đã hoàn tất; nhóm đã tải ZIP kết quả và ZIP checkpoint riêng. BiGRU và PhoBERT đều chạy trên Tesla T4. Kết quả đã được đối chiếu với ID validation, lịch sử chọn checkpoint và các log kết thúc thành công.”

**Thao tác:** Chỉ `environment`, `device` và `history` trong [BiGRU run JSON](evidence/gru_run.json) hoặc [PhoBERT run JSON](evidence/phobert_run.json). Không mở checkpoint dung lượng lớn trong buổi trình bày.

## Slide 11 — Công việc tiếp theo · 40 giây

**Lời nói:**

“Bước tiếp theo là hoàn tất study learning rate của PhoBERT với ba giá trị: 1e-5, 2e-5 và 3e-5. Nhóm giữ các cấu hình khác cố định và so sánh trên validation. Hiện chưa có kết quả mới của study được nhập và kiểm chứng, nên nhóm chưa kết luận learning rate nào tốt nhất.

Sau đó, nhóm sẽ đọc lỗi BiGRU và PhoBERT, cập nhật insight, đồng thời làm rõ với thầy/cô việc sử dụng split được cấp có đáp ứng yêu cầu stratification trong đề hay cần protocol bổ sung.”

**Thao tác:** Mở [notebook 05](../../notebooks/05_colab_phobert_tuning.ipynb) hoặc [thiết kế tuning](../../docs/phobert_tuning.md). Không dùng kết quả study cũ cho lượt mới.

## Slide 12 — Hướng Final và kết thúc · 35 giây

**Lời nói:**

“Cho Final, nhóm dự kiến kiểm tra câu hỏi: một threshold chung 0,5 có phù hợp với tất cả 27 nhãn không? Nhóm sẽ so sánh ngưỡng cố định, ngưỡng chung được tune và ngưỡng riêng từng nhãn, trên cùng một neural model và giữ nguyên trọng số backbone.

Tất cả ngưỡng được chọn trên validation, khóa lại rồi mới đánh giá test đúng một lần. Đến hiện tại, nhóm đã có pipeline dữ liệu và ba baseline có minh chứng; phần còn lại là hoàn thiện tuning và phân tích lỗi. Nhóm em xin nhận góp ý của thầy/cô.”

**Thao tác:** Dừng ở kế hoạch và phần còn thiếu; không tuyên bố đã hoàn thành Final.

## Câu hỏi thường gặp — dùng khi được hỏi

### Vì sao 10 epoch cho BiGRU, 4 epoch cho PhoBERT?

“BiGRU học từ đầu, còn PhoBERT đã có backbone pretrained. Nhóm đặt ngân sách ban đầu khác nhau và chọn checkpoint theo validation, chưa chứng minh các số epoch này là tối ưu. Trong lượt mới, best epoch còn ở cuối nên cần xem learning curve nếu muốn mở rộng thí nghiệm.”

### TF-IDF có epoch không?

“TF-IDF không train theo epoch; nó tính đặc trưng từ dữ liệu train. Logistic Regression phía sau dùng solver lặp đến khi hội tụ hoặc đạt `max_iter=1000`; max_iter không tương đương số epoch của neural model.”

### Vì sao chọn threshold 0,5?

“Đó là mốc ban đầu chung cho ba baseline. Threshold này chưa được tối ưu cho Macro-F1 và dữ liệu mất cân bằng. Nhóm đã khảo sát độ nhạy trên validation; phần Final sẽ so sánh các chiến lược threshold có hệ thống.”

### Đã fine-tune hay chỉ chạy model có sẵn?

“Nhóm đã fine-tune toàn bộ trọng số PhoBERT cùng classification head bằng dữ liệu train. Fine-tuning là học lại trọng số cho nhiệm vụ; hyperparameter tuning là so sánh các cấu hình như learning rate. Phần đầu đã hoàn tất baseline, phần study mới chưa có kết luận được kiểm chứng.”

### Có điều chỉnh trọng số theo từng mẫu không?

“Chưa có sample weighting hay class weighting trong BCE hiện tại. Trọng số của model được cập nhật theo từng batch bằng gradient; checkpoint được chọn theo Macro-F1 validation sau epoch. Đây là các việc khác nhau.”

### Đã dùng test chưa?

“Nhóm có kiểm tra schema, thống kê mô tả và dò trùng để audit dữ liệu. Chưa dùng test để chọn learning rate, epoch, checkpoint hoặc threshold; chưa báo cáo metric test.”

### Paper có dùng stratification không?

“Paper mục 5.1 chỉ mô tả tỷ lệ 8:1:1. Nhóm chưa tìm thấy thuật toán chia tập nên chưa thể xác nhận có hay không có stratification. Nhóm giữ split được cấp và đã audit phân bố; xin làm rõ yêu cầu này với thầy/cô.”

### Tại sao điểm PhoBERT mới khác lần trước?

“Lượt cũ có Macro-F1 khoảng 0,4393; lượt mới là 0,4679. Đây là hai lượt huấn luyện riêng. Cùng seed không bảo đảm kết quả giống tuyệt đối giữa môi trường hoặc phép tính GPU. Nhóm chưa cô lập nguyên nhân chênh lệch, nên không khẳng định do một thay đổi cụ thể và không coi hai lượt là kiểm định nhiều seed.”

### TF-IDF tune ngưỡng đạt 0,4844, sao lại nói PhoBERT tốt nhất?

“PhoBERT tốt nhất trong bảng ba baseline cùng threshold 0,5. TF-IDF 0,4844 đã được chọn ngưỡng trên validation; cần so sánh các chiến lược decoding tương ứng và đánh giá test theo giao thức đã khóa trước khi kết luận mô hình tốt nhất sau tuning.”

## Checklist cho người trình bày

- Mở bảng trong thư mục `evidence/` đi kèm script này; bộ baseline gốc ở ngoài vẫn được giữ nguyên.
- Cập nhật các số/trạng thái trên slide cũ trước khi dùng với lời thoại mới; không ghép score lượt mới với style delta lượt cũ.
- Nói “điểm phần trăm” khi lấy chênh lệch F1 nhân 100; không gọi là phần trăm tăng tương đối.
- Phân biệt việc đã xuất CSV lỗi với việc đã đọc và giải thích lỗi.
- Không dùng từ “đã tối ưu” cho epoch, learning rate hoặc threshold chưa được kiểm chứng.
- Nếu chưa có artifacts study 05, trình bày đúng là chưa có kết luận được kiểm chứng; không suy đoán winner từ một vài epoch.
