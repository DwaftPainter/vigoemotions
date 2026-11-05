# Chạy ViGoEmotions trên Kaggle — 20 epoch

Workflow chính là [notebook 08 cho 27 nhãn](../notebooks/08_kaggle_equal_20ep_27_labels.ipynb) và [notebook 09 cho 28 nhãn](../notebooks/09_kaggle_equal_20ep_28_labels.ipynb). Mỗi bản chạy TF-IDF, BiGRU và PhoBERT; neural đủ 20 epoch. Dùng chung configs/kaggle_equal_20ep.json, seed 42, split và tiêu chí emotion27 Macro-F1 tại 0,5.

Hai lượt đã hoàn tất. Xem [kết quả chính đã kiểm chứng](../reports/kaggle_equal20_2026-11-05/README.md), [profile và lệnh tái lập](reproducibility_profiles.md), [lời dẫn thuyết trình](midterm_presentation_2026-11-05.md). Không cần train lại để trình bày kết quả hiện có.

## Nếu cần chạy một lượt mới

1. Chạy `.venv/bin/python scripts/package_kaggle_equal_epochs.py` tại project root để tạo vigoemotions_kaggle_equal_20ep.zip.
2. Tạo Dataset code chứa ZIP và Dataset Private chứa train.csv, val.csv, test.csv với cột id,text,labels. Hai notebook dùng chung hai Dataset này.
3. Import notebook 08 hoặc 09, bật GPU + Internet, giữ Private. Chạy tuần tự hoặc Save Version → Save & Run All. Mỗi notebook chỉ chạy ba model của bộ nhãn tương ứng.
4. Chờ hoàn tất rồi tải ZIP kết quả từ Output và tải notebook executed kèm output. Save & Run All tạo session mới và train lại từ đầu; Quick Save không chạy lại code.
5. ZIP kết quả là vigoemotions_kaggle_equal20_27labels_private_results.zip hoặc bản 28labels. Không đưa checkpoint/vocabulary lên repo công khai.

Hướng dẫn cấu hình, resume, validation và archive chi tiết tại [giao thức equal20](kaggle_equal_epochs.md). Các workflow 04/06 và profile ít epoch trước đây là lịch sử, không làm nguồn điểm chính.
