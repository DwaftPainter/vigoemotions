# ViGoEmotions — Ôn Final oral defence

Nhánh **`study/final-oral-defence`** dùng để ôn RQ1 về ngưỡng dự đoán và RQ4 về nhiễu văn bản trên PhoBERT 28 đầu ra. Bản Final đầy đủ nằm trên nhánh **`submission/ati-final-rq4`**. [Report Word](https://github.com/DwaftPainter/vigoemotions/blob/cb0ea5c/reports/final_report_ati.docx)

## Bắt đầu ôn

| Thứ tự | Nội dung | Markdown | PDF |
| --- | --- | --- | --- |
| 00 | Tra cứu nhanh: kết luận, số cần nhớ, lỗi dễ nhầm | [Đọc](docs/oral_defence/00_quick_revision_vi.md) | [PDF](docs/oral_defence/00_quick_revision_vi.pdf) |
| 01 | ML/NLP, ba mô hình, sigmoid/BCE và metrics | [Đọc](docs/oral_defence/01_foundations_vi.md) | [PDF](docs/oral_defence/01_foundations_vi.pdf) |
| 02 | Thiết kế và kết quả RQ1/RQ4, Neutral, tiers/cardinality và code map | [Đọc](docs/oral_defence/02_protocol_results_vi.md) | [PDF](docs/oral_defence/02_protocol_results_vi.pdf) |
| 03 | 34 câu Q&A và câu trả lời ngắn tiếng Anh | [Đọc](docs/oral_defence/03_questions_answers_vi.md) | [PDF](docs/oral_defence/03_questions_answers_vi.pdf) |
| 04 | Tính tay, trace decoder và hai buổi mock defence | [Đọc](docs/oral_defence/04_practice_vi.md) | [PDF](docs/oral_defence/04_practice_vi.pdf) |

**Có 30 phút:** đọc 00, tự nói mở đầu, trả lời Q11, Q20, Q27, Q33 và Q34. **Có 2 giờ:** đọc 01–02, chọn 10 câu Q&A và mở code theo bản đồ trong tài liệu 02. **Có hai buổi:** luyện theo 04, đổi vai và ghi những chỗ chưa giải thích được. Dùng lịch giảng viên thông báo để sắp xếp các buổi luyện.

## Kết quả cần nhớ

Final test **all28** Macro-F1: fixed **0.589964**, global 0.4 **0.591254**, per-label **0.585415**. Global tăng **0.1289 pp**, per-label giảm **0.4549 pp** so với fixed. Fixed có Hamming Loss thấp nhất. Checkpoint là epoch15 từ training đủ 20 epoch, seed42; checkpoint criterion gốc emotion27 khác objective threshold all28.

RQ4 ở fixed 0.5 cho Macro-F1 test **0.589964** trên văn bản gốc, **0.329749** sau khi bỏ dấu và **0.550578** khi tạo lỗi gõ QWERTY. So với gốc, hai mức giảm là **26.02** và **3.94** điểm phần trăm. Hai phép biến đổi có mức độ khác nhau. [Kiểm chứng RQ4](reports/rq4_noise/verification.json)

Đọc mục 5.3 của [report Final đầy đủ](https://github.com/DwaftPainter/vigoemotions/blob/cb0ea5c/reports/final_report_ati.pdf) khi cần ôn RQ4. [Bản RQ1 trước khi làm extension](reports/final_report.pdf) và [report Midterm tiếng Anh](reports/midterm_report.pdf) vẫn dùng để ôn ngưỡng và baseline. [Đề ViGoEmotions](docs/02-vigoemotions-vietnamese-emotion.pdf) · [Hướng dẫn nộp ATI](https://github.com/DwaftPainter/vigoemotions/blob/cb0ea5c/docs/ATI_Submission_Guide.pdf)

## Cấu trúc nhánh đã dọn

- `docs/oral_defence/`: năm bộ ôn Markdown/PDF, templates để sửa và verification.
- `reports/final28/`: scores, bảng, hình, ngưỡng/lock và case notes đã hoàn thành; đọc qua [index](reports/final28/README.md).
- `reports/rq4_noise/`: bảng, hình và xác minh RQ4 dùng trong bộ ôn. Đầu ra dự đoán đầy đủ nằm trên nhánh nộp bài.
- `reports/kaggle_equal20_2026-11-05/`: evidence baseline Midterm cần để giải thích model/checkpoint; [index](reports/kaggle_equal20_2026-11-05/README.md).
- `scripts/`: source cần đọc khi bảo vệ và builder tài liệu ôn. `configs/`, `splits/` giữ protocol và IDs gốc.
- `tests/`: checks dùng cho nội dung/PDF ôn và decoder được giữ nguyên. Đây là phạm vi riêng, không phải full suite của nhánh nghiên cứu.

[Cleanup manifest](docs/oral_defence/cleanup_manifest.json) ghi các file đã bỏ khi tạo nhánh ôn. [Source manifest](docs/oral_defence/source_manifest.json) ghi hash file RQ1 được giữ và các nguồn RQ4 bổ sung từ commit `cb0ea5c`.

## Dựng lại PDF ôn

Cài các dependency dựng tài liệu trong `requirements-study.txt`, rồi chạy từ project root:

```bash
.venv/bin/python scripts/build_oral_defence.py
.venv/bin/python -m unittest discover -s tests -p test_oral_defence.py -v
```

Sửa nội dung tại `docs/oral_defence/templates/`, không sửa trực tiếp các bảng số liệu sinh ra. Builder dùng CSV/JSON đã lưu, kiểm source hashes và xuất năm cặp Markdown/PDF cùng [verification](docs/oral_defence/verification.json). Không cần torch, GPU, raw CSV hoặc private weights để dựng tài liệu; builder không train hay inference. `requirements.txt` / `requirements-colab.txt` giữ từ nguồn đầy đủ để tra môi trường nghiên cứu, không phải yêu cầu tối thiểu của notes.

Protocol/config frozen có status trước execution; trạng thái hoàn thành xem artifacts Final. Không chỉnh ngưỡng, rerun inference hoặc tự suy thông tin nhóm từ tên Git author khi ôn.

Với môi trường nghiên cứu đã có numpy/pandas/scikit-learn theo `requirements.txt`, chạy `.venv/bin/python -m unittest discover -s tests -v` để kiểm cả decoder tests được giữ và ba notes tests. Đây là 32 tests áp dụng cho nhánh ôn, không phải bộ đầy đủ của nhánh nguồn.

## Quay lại nguồn đầy đủ

Dùng nhánh `feature/final28-report` để làm submission hoặc chạy workflow đầy đủ. Commit `289f2fb` giữ toàn bộ nội dung trước khi dọn; nhánh nguồn không bị thay đổi. Tài liệu ôn có thể đọc offline; công việc còn lại là nhóm luyện nói và xác nhận lịch, ngôn ngữ, thời lượng, phân công thật.
