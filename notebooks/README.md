# Notebook đang dùng

| Notebook | Vai trò |
| --- | --- |
| [01](01_dataset_validation.ipynb) | Kiểm tra dữ liệu và nhãn |
| [02](02_exploratory_analysis.ipynb) | EDA |
| [07](07_split_distribution_audit.ipynb) | Audit split được cấp |
| [08 template](08_kaggle_equal_20ep_27_labels.ipynb), [executed](08_kaggle_equal_20ep_27_labels_executed.ipynb) | Baseline 27 nhãn để đối chiếu lịch sử |
| [09 template](09_kaggle_equal_20ep_28_labels.ipynb), [executed](09_kaggle_equal_20ep_28_labels_executed.ipynb) | Nguồn checkpoint PhoBERT 28 epoch 15 dùng cho Final |
| [10 validation](10_kaggle_final_validation.ipynb) | Khôi phục PhoBERT 28 → tái tạo validation → fit global/28 ngưỡng per-label → khóa; không inference test |

Notebook 03–06 đã chuyển sang [archive/notebooks](../archive/notebooks/); xem [mục lục archive](../archive/README.md). Các notebook executed 08/09 giữ nguyên byte/output để kiểm chứng provenance.

Thiết kế nằm ở [protocol Final](../docs/final_protocol.md), lệnh chạy tại [final_execution.md](../docs/final_execution.md). Notebook 10 cần actual GPU inference để tạo bằng chứng; unit tests không thay cho bước này. Mở notebook training 08/09 sẽ train lại, không phải chạy threshold tuning trên saved probabilities.
