# ViGoEmotions — Ôn Final oral defence

Nhánh **`study/final-oral-defence`** dành cho ôn kiến thức và luyện bảo vệ Final RQ1 trên PhoBERT 28 outputs. Bản nghiên cứu đầy đủ ở nhánh **`feature/final28-report`**, commit **`289f2fb`**. Nhánh nộp bài sẽ tạo sau từ nguồn đầy đủ, không từ bản đã dọn này.

## Bắt đầu ôn

| Thứ tự | Nội dung | Markdown | PDF |
| --- | --- | --- | --- |
| 00 | Tra cứu nhanh: kết luận, số cần nhớ, lỗi dễ nhầm | [Đọc](docs/oral_defence/00_quick_revision_vi.md) | [PDF](docs/oral_defence/00_quick_revision_vi.pdf) |
| 01 | ML/NLP, ba mô hình, sigmoid/BCE và metrics | [Đọc](docs/oral_defence/01_foundations_vi.md) | [PDF](docs/oral_defence/01_foundations_vi.pdf) |
| 02 | Protocol, kết quả, Neutral, tiers/cardinality và code map | [Đọc](docs/oral_defence/02_protocol_results_vi.md) | [PDF](docs/oral_defence/02_protocol_results_vi.pdf) |
| 03 | 30 câu Q&A và câu trả lời ngắn tiếng Anh | [Đọc](docs/oral_defence/03_questions_answers_vi.md) | [PDF](docs/oral_defence/03_questions_answers_vi.pdf) |
| 04 | Tính tay, trace decoder và hai buổi mock defence | [Đọc](docs/oral_defence/04_practice_vi.md) | [PDF](docs/oral_defence/04_practice_vi.pdf) |

**Có 30 phút:** đọc 00, tự nói opening, trả lời Q11/Q20/Q21/Q26 và làm bài metric trong 04. **Có 2 giờ:** đọc 01–02, chọn 10 câu Q&A và mở code theo map. **Có hai buổi:** luyện theo 04, đổi vai và ghi những chỗ chưa giải thích được. Đây là lịch ôn gợi ý; thời lượng và quy định oral chính thức chưa được xác nhận.

## Kết quả cần nhớ

Final test **all28** Macro-F1: fixed **0.589964**, global 0.4 **0.591254**, per-label **0.585415**. Global tăng **0.1289 pp**, per-label giảm **0.4549 pp** so với fixed. Fixed có Hamming Loss thấp nhất. Checkpoint là epoch15 từ training đủ 20 epoch, seed42; checkpoint criterion gốc emotion27 khác objective threshold all28.

Đọc [paper Final](reports/final_report.pdf) / [Markdown](reports/final_report.md) khi cần lập luận đầy đủ. [Report Midterm tiếng Anh](reports/midterm_report.pdf) / [Markdown](reports/midterm_report.md) chỉ cung cấp bối cảnh baseline; các kế hoạch pending trong đó mô tả thời điểm Midterm. [Đề ViGoEmotions](docs/02-vigoemotions-vietnamese-emotion.pdf) và [student guide](docs/student-guide.pdf) là nguồn yêu cầu.

## Cấu trúc nhánh đã dọn

- `docs/oral_defence/`: năm bộ ôn Markdown/PDF, templates để sửa và verification.
- `reports/final28/`: scores, bảng, hình, ngưỡng/lock và case notes đã hoàn thành; đọc qua [index](reports/final28/README.md).
- `reports/kaggle_equal20_2026-11-05/`: evidence baseline Midterm cần để giải thích model/checkpoint; [index](reports/kaggle_equal20_2026-11-05/README.md).
- `scripts/`: source cần đọc khi bảo vệ và builder tài liệu ôn. `configs/`, `splits/` giữ protocol và IDs gốc.
- `tests/`: checks dùng cho nội dung/PDF ôn và decoder được giữ nguyên. Đây là phạm vi riêng, không phải full suite của nhánh nghiên cứu.

Archive, extensions chưa hoàn tất, output model cũ, DOCX và công cụ dựng/nộp không dùng để ôn đã bỏ khỏi nhánh này. Một file snapshot hashes của prototype27 được giữ vì protocol frozen có link tới nó. [Cleanup manifest](docs/oral_defence/cleanup_manifest.json) ghi từng file đã bỏ; [source manifest](docs/oral_defence/source_manifest.json) ghi hash file gốc được giữ. File private đang Git ignore không bị xóa và không thuộc nội dung nhánh chia sẻ.

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
