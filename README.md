# ViGoEmotions — ATI Final 2026

Midterm đã hoàn tất. Final tập trung RQ1 trên **PhoBERT 27 nhãn, checkpoint epoch 14**: fixed 0,5, global tuned và per-label tuned; chọn ngưỡng trên validation, khóa trước test. Protocol đã ghi, fitted thresholds và kết quả test Final chưa có.

- [Protocol Final](docs/final_protocol.md), [config](configs/final_thresholds.json), [train-only tiers](docs/final_label_tiers.csv).
- [Checklist việc cần làm](tasks/todo.md) và [kế hoạch Final/oral defence](tasks/plan.md).
- [Mục lục archive](archive/README.md): notebook 03–06 và tài liệu chuẩn bị cũ đã được chuyển khỏi thư mục làm việc chính.

## Baseline midterm dùng cho Final

**Kết quả chính: hai lượt Kaggle 08/09 đã hoàn tất, BiGRU và PhoBERT đều 20 epoch cho cả 27/28 nhãn.** Báo cáo dùng duy nhất sáu kết quả này; các lượt ít epoch là lịch sử. Tất cả metric mô hình là validation tại ngưỡng 0,5, seed 42; test giữ cho Final.

| Mô hình | Macro-F1 27 nhãn | Macro-F1 28 nhãn |
| --- | --- | --- |
| TF-IDF + OvR LR | 0.2388 | 0.2310 |
| BiGRU | 0.4220 | 0.4086 |
| PhoBERT | 0.5974 | 0.5783 |

Khi chiếu model 28 về cùng 27 cảm xúc, PhoBERT đạt **0,583580**, thấp hơn model 27 (**0,597410**) khoảng **1,383 điểm phần trăm**. Lượt đo này chưa cho thấy thêm Neutral cải thiện 27 cảm xúc. Neutral F1 của PhoBERT là **0,4369** (45 TP, 67 FP, 49 FN). Một seed chưa đủ kết luận nhân quả hoặc ý nghĩa thống kê.

## Mở nội dung để thuyết trình

- [Báo cáo PDF](reports/midterm_report.pdf) / [Markdown](reports/midterm_report.md).
- [PDF: 15 slide, bảng Q1–Q4 tại slide 10–11](reports/midterm_slides.pdf) / [nội dung và speaker notes](reports/midterm_slides.md).
- [Script tiếng Anh 8–9 phút + Q&A](docs/midterm_presentation_en.md) / [PDF để luyện nói](docs/midterm_presentation_en.pdf). Bản giải thích tiếng Việt vẫn ở [lời dẫn ôn tập](archive/docs/midterm_presentation_2026-11-05.md).
- [Bảng sáu model, Q1–Q5 và evidence](reports/kaggle_equal20_2026-11-05/README.md).
- [Cẩm nang giải thích ba mô hình](docs/VIGOEMOTIONS_3_MODELS_STUDY_GUIDE_VI.pdf) / [Markdown](docs/VIGOEMOTIONS_3_MODELS_STUDY_GUIDE_VI.md).
- [Trạng thái trước nộp](archive/docs/midterm_readiness_2026-11-05.md), [thông tin nhóm cần bổ sung](docs/team_information.md).

## Nguồn chạy và kiểm chứng

- [Notebook 08 template](notebooks/08_kaggle_equal_20ep_27_labels.ipynb) / [08 executed, giữ output](notebooks/08_kaggle_equal_20ep_27_labels_executed.ipynb).
- [Notebook 09 template](notebooks/09_kaggle_equal_20ep_28_labels.ipynb) / [09 executed, giữ output](notebooks/09_kaggle_equal_20ep_28_labels_executed.ipynb).
- Hai ZIP private ở project root: `vigoemotions_kaggle_equal20_27labels_private_results.zip` và bản `28labels`; không commit checkpoint/vocabulary.
- [Giao thức Kaggle](docs/kaggle_equal_epochs.md), [profile và lệnh tái lập](docs/reproducibility_profiles.md), [provenance](reports/kaggle_equal20_2026-11-05/provenance.json).

Đã tính lại metric từ cả sáu NPZ, xác minh cùng ID/targets/label order, đủ history 20 epoch, checkpoint selection và file trọng số trong ZIP. Hai notebook đều chạy đủ 12 cell code, không có error output. Source/config/CSV của lượt 28 có một số checksum khác do CRLF; từng hash đã được chứng minh khớp biến thể LF/CRLF chính xác của cùng nội dung local. Manifest gốc được giữ nguyên.

## Dữ liệu và preprocessing

20.664 bình luận nguồn; split 16.531/2.066/2.067. Loại một mẫu train trùng test → 16.530 mẫu train thực dùng. Xin quyền dữ liệu từ dataset ViGoEmotions và đặt CSV `id,text,labels` tại `data/train.csv`, `val.csv`, `test.csv`. Không công khai CSV/bình luận gốc, vocabulary hoặc checkpoint. Git ignore data và ZIP riêng tư.

Giữ split được cấp, chỉ fit vocabulary/IDF trên train. Chuẩn hóa NFC/khoảng trắng, giữ dấu và phủ định. 27 output bỏ Neutral; 28 output giữ ID 27 và mixed annotation. [Mapping](docs/label_mapping.md), [EDA](docs/dataset_check.md) và [audit split 07](notebooks/07_split_distribution_audit.ipynb) vẫn dùng cùng dữ liệu. Audit phân bố không chứng minh thuật toán stratification gốc; còn xác nhận yêu cầu split với giảng viên.

## Tái tạo nội dung, không train lại

Trong môi trường local có dependencies theo `requirements.txt`:

```bash
.venv/bin/python scripts/import_equal20_results.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/build_study_guide.py
```

Tạo riêng slide/script tiếng Anh từ evidence đã kiểm chứng: `.venv/bin/python scripts/build_equal20_presentation.py`. Lệnh build report đầy đủ cũng giữ slide bằng tiếng Anh.

Bản thiết kế hiện tại thay slide 10 bằng bảng Q1/Q2 và slide 11 bằng bảng Q3/Q4 theo yêu cầu; 13 slide còn lại và toàn bộ lời nói đã chốt được giữ nguyên. Snapshot nội dung gốc là `reports/kaggle_equal20_2026-11-05/slides.json`. Deck có 15 slide, không có trang bảng lặp ở cuối; vị trí bảng được ghi chú trong script. Renderer không ghi lại evidence; layout tự sinh ở `reports/midterm_slide_design.json` và kiểm chứng ở `reports/midterm_presentation_verification.json`.

Import cần hai ZIP, notebook executed và CSV nguồn. Evidence/tổng hợp mới nằm riêng tại `reports/kaggle_equal20_2026-11-05/`; không ghi đè bảng model các lượt cũ. Build report thiếu evidence mới sẽ dừng, không tự quay về số cũ. Cẩm nang kiểm tra nguyên văn các đoạn trích source trước khi render.

Để train lượt mới: `.venv/bin/python scripts/package_kaggle_equal_epochs.py`, upload ZIP code và ba CSV nguồn thành hai Dataset Private, import notebook 08 hoặc 09, bật GPU + Internet. Chạy theo [hướng dẫn Kaggle](docs/kaggle_equal_epochs.md). Không cần train lại để dùng các kết quả hiện có.

## Giới hạn và việc còn lại

Điền tên/MSSV/phân công; xác nhận sáu ca đọc lỗi có hỗ trợ AI; làm rõ việc chấp nhận official split. Chưa có metric test hoặc sweep learning rate. Final giữ backbone neural cố định, so sánh fixed/global/per-label thresholds, chọn trên validation rồi khóa trước khi chấm test. Một seed và style heuristic còn hạn chế.

Notebook 01/02/07 phục vụ kiểm tra dữ liệu, EDA và audit split; notebook 03–06 nằm trong [archive/notebooks](archive/notebooks/), config 25/20 cũ trong [archive/configs](archive/configs/). Các snapshot và tài liệu chuẩn bị cũ nằm trong [archive/](archive/README.md). Config/run JSON và bảng model cũ còn được script/test tham chiếu được giữ nguyên vị trí; kết quả chính chỉ lấy từ thư mục Kaggle 20 epoch. [Mục lục reports](reports/README.md).
