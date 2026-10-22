# Xác suất validation — NPZ

[← Mục lục reports](../README.md)

| File | Mô hình | Kích thước ma trận |
| --- | --- | --- |
| [tfidf_lr_val.npz](tfidf_lr_val.npz) | TF-IDF + OvR Logistic Regression | 2.066 × 27 |
| [gru_val.npz](gru_val.npz) | BiGRU | 2.066 × 27 |
| [phobert_val.npz](phobert_val.npz) | PhoBERT | 2.066 × 27 |

Đây là đầu ra **liên tục trước khi áp threshold**, không phải trọng số mô hình. Lưu chúng để tính lại metric, đối chiếu lỗi và khảo sát decoder mà không huấn luyện lại. Chưa có NPZ test.

## Các mảng bên trong

| Key | Shape | Ý nghĩa |
| --- | --- | --- |
| `sample_id` | `(2066,)` | ID theo đúng thứ tự hàng; dùng join với dữ liệu nội bộ nếu được cấp quyền |
| `true_label_vector` | `(2066, 27)` | Ground truth multi-hot sau xử lý neutral/nhãn lặp |
| `probability_vector` | `(2066, 27)` | Xác suất từng nhãn, hữu hạn trong [0,1] |
| `labels` | `(27,)` | Tên nhãn theo đúng thứ tự cột; không tự sắp alphabet |
| `label_cardinality` | `(2066,)` | Tổng nhãn dương của ground truth từng hàng |

TF-IDF đang lưu float64; neural đang lưu float32. Khi tính LRAP, evaluator chuyển sang float64 để tránh sai khác tích lũy rất nhỏ giữa các phiên bản thư viện; không đổi thứ hạng hoặc giá trị đã lưu trong NPZ.

## Đọc từ project root

```python
import numpy as np

with np.load('reports/predictions/phobert_val.npz', allow_pickle=False) as saved:
    sample_ids = saved['sample_id']
    y_true = saved['true_label_vector']
    probabilities = saved['probability_vector']
    predictions = probabilities >= 0.5
    label_names = saved['labels']
```

Các file không chứa bình luận nguồn. Dù vậy, ID/nhãn vẫn cần chia sẻ theo điều khoản dữ liệu. Không thêm raw text, vocabulary hoặc checkpoint vào thư mục này. Muốn chạy mô hình trên văn bản mới cần checkpoint và đúng tiền xử lý, không thể thay bằng NPZ.
