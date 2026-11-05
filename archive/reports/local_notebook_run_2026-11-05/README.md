# Lượt chạy lại notebook local — 05/11/2026

Đã chạy toàn bộ cell của notebook 01, 02, 03 và 07 bằng `.venv` của project. Các notebook giữ output tổng hợp an toàn; không xuất bình luận nguồn. Notebook 03 thực sự fit lại TF-IDF trên CPU, không train BiGRU/PhoBERT hoặc chấm test. Các dòng neural trong bảng của notebook 03 vẫn là kết quả lịch sử để tham khảo.

Lượt chạy thực hiện trong thư mục project tách riêng để giữ nguyên các bảng/JSON/xác suất baseline đã dùng cho midterm. Các artifact của lượt local được sao chép vào thư mục này, kèm `execution_manifest.json` ghi thời điểm, số cell/output, hash notebook và hash dữ liệu/kết quả lịch sử được bảo toàn.

- `tables/`, `figures/`: EDA và đánh giá TF-IDF từ lượt local.
- `predictions/tfidf_lr_val.npz`: xác suất validation mới, không có text.
- `tfidf_lr_run.json`: cấu hình, phiên bản và thời gian fit mới.
- `split_audit/`: bằng chứng phân bố/tính toàn vẹn từ notebook 07.

Notebook 04/05/06 là workflow Colab/Kaggle, cần GPU, dependency hoặc thao tác cloud; không được chạy lại toàn bộ trên local trong lượt này. Không suy ra các notebook đó đã có kết quả chạy mới.
