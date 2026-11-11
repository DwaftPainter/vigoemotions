# Chạy pipeline Final RQ1

Thiết kế: [protocol v1](final_protocol.md), [config](../configs/final_thresholds.json). Pipeline không train model. Mọi lệnh mặc định ghi dưới `reports/final/`; checkpoint và VnCoreNLP assets nằm trong `data/final/` đã Git ignore.

## Trình tự

| Lệnh | Công việc | Cần GPU/thư viện neural? |
| --- | --- | --- |
| `verify` | Kiểm source hashes, fixed IDs/targets, Macro-F1 nguồn và train-only tiers | Không |
| `restore` | Kiểm SHA-256 ZIP rồi trích đúng checkpoint/segmentation assets private | Không |
| `reproduce-val` | Nạp frozen checkpoint, inference đủ validation, kiểm sai lệch ≤1e-4 và quyết định fixed giống hệt | Có thư viện neural; GPU nên dùng để khớp môi trường nguồn |
| `fit` | Tìm global/per-label thresholds từ NPZ nguồn, xuất ba decoder và bảng validation | Không, nhưng cần reproduction thật đã PASS |
| `lock` | Khóa hashes protocol/config/implementation/source predictions/thresholds/reproduction | Không |
| `infer-test` | Sau khi khóa, inference test đúng một lượt, lưu scores rồi mới decode | Có; không nằm trong notebook validation mặc định |
| `report` | Tái sinh bảng từ saved probabilities/decoder outputs; không inference lại | Không |

Ví dụ local với môi trường `.venv` đã có dependencies tương ứng:

```bash
.venv/bin/python scripts/final_thresholds.py verify
.venv/bin/python scripts/final_thresholds.py restore
.venv/bin/python scripts/final_thresholds.py reproduce-val --device cuda
.venv/bin/python scripts/final_thresholds.py fit
.venv/bin/python scripts/final_thresholds.py lock
```

Local hiện có thể thiếu torch/transformers/py-vncorenlp. `verify`, `restore` và thuật toán decoder không import các thư viện neural; `reproduce-val` báo lỗi dependency rõ ràng thay vì tạo bằng chứng giả. Không có file reproduction verified thì `fit`/`lock`/`infer-test` đều dừng.

## Chạy validation trên Kaggle

1. Tạo code ZIP: `.venv/bin/python scripts/package_final.py`.
2. Attach ba nguồn Private: `vigoemotions_final_code.zip`, ZIP kết quả 27 nhãn nguyên byte và CSV có quyền truy cập. ZIP weights cần còn nguyên file để đối chiếu hash; notebook cũng chấp nhận cùng byte ZIP với đuôi `.bin` nếu nền tảng tự giải nén ZIP.
3. Import [notebook 10](../notebooks/10_kaggle_final_validation.ipynb), bật GPU và Internet. Notebook chỉ copy **train/val** vào workspace, dùng split IDs đã có trong code ZIP.
4. Chạy tuần tự verify → restore → reproduce-val → fit; đọc bảng validation, sau đó cell lock. Notebook không có cell gọi `infer-test`.
5. Tải thư mục `reports/final/` về để review và giữ workspace/model assets private. Giữ cùng implementation/config sau khi khóa; chỉnh code/protocol sau lock sẽ làm gate stale.

Trong Kaggle, dùng `sys.executable` thay `.venv/bin/python`. `--archive`, `--data-dir`, `--private-dir` và `--output-dir` cho phép chỉ rõ nguồn/đích; mọi hash và contract vẫn phải khớp. Output trong repo chỉ được nằm dưới `reports/final`, private assets chỉ dưới ignored `data/`; một thư mục reproduction ngoài repo cũng được phép.

## Test held-out sau khi đã review ngưỡng khóa

Chỉ chạy bước này khi đã sẵn sàng đánh giá held-out theo protocol. Cần `test.csv` private đúng checksum gốc trong `--data-dir`, cùng checkpoint và segmentation assets đã dùng tái tạo validation.

```bash
.venv/bin/python scripts/final_thresholds.py infer-test --device cuda
.venv/bin/python scripts/final_thresholds.py report
```

Test kiểm pre-test lock trước khi đọc test text/targets, đồng thời giữ nguyên package versions/Python và device đã dùng tái tạo validation. Một lượt hoàn tất sẽ chặn lần inference tiếp theo. Nếu lỗi kỹ thuật, giữ `test_attempt.json`; `--resume-failed` chỉ tiếp tục với cùng lock và assets. Nếu continuous test scores đã lưu xong thì resume dùng lại chúng, không gọi model lần nữa. Trạng thái chạy dở do process bị kill cần được điều tra; không tự xóa gate hoặc đổi ngưỡng để chạy lại.

## Artifacts và cách đọc

- `validation_reproduction.json`: chỉ có sau inference thật khớp toàn bộ NPZ; ghi packages/device/assets/source hashes.
- `thresholds.json`: global scalar, 27 cutoffs, support/fallback, grid sweep và source bindings. Không sửa tay; đọc lại được đối chiếu với phép tối ưu validation.
- `pre_test_lock.json`: bằng chứng mọi lựa chọn đã cố định trước test.
- `predictions/val_reproduced.npz`: scores tái tạo từ model. `predictions/val.npz`: NPZ nguồn cùng `pred_fixed`, `pred_global`, `pred_per_label`.
- `predictions/test_probabilities.npz`: continuous scores lưu trước decoding; `predictions/test.npz`: cùng scores và ba bộ dự đoán.
- `tables/{val,test}_{comparison,per_label,frequency_tiers,cardinality}.csv`: các bảng metric/support và delta so với fixed. Nhóm cardinality rỗng có n=0, metric để trống.
- `tables/threshold_profile.csv`, `global_sweep.csv`, `generalization_gap.csv`: profile ngưỡng, grid và gap val–test; gap chỉ có sau test.

Decoder chuyển scores sang float64 trước so sánh để scalar/vector thresholds có cùng hành vi ở boundary; không đổi giá trị scores gốc được lưu. `p >= threshold`, hòa F1 chọn cutoff cao hơn, nhãn thiếu positive validation dùng 0,5. LRAP giữ nguyên khi chỉ đổi cutoff.

`report` không thay byte của prediction NPZ hay fitted/lock files. Lệnh cần authorized train/val để kiểm source bindings; test breakdown chỉ đọc saved test NPZ đã hoàn tất, không đọc lại CSV test để tuning.

## Kiểm thử

```bash
.venv/bin/python -m unittest discover -s tests -p test_final_thresholds.py -v
.venv/bin/python -m unittest discover -s tests -v
```

Fixture inference trong unit tests chỉ dùng dữ liệu giả ở thư mục tạm, không phải bằng chứng tái tạo PhoBERT thật. Full suite có hai test neural cũ cần `torch`; thiếu dependency phải báo rõ, không skip hoặc đổi assertion để có xanh.

API references: [PyTorch state-dict serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html), [Transformers 4.57.3 Auto Classes](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/auto), [scikit-learn F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).
