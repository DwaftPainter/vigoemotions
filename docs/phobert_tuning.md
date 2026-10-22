# PhoBERT — Điều chỉnh learning rate trước Midterm

## Trạng thái và mục tiêu

Đã fine-tune PhoBERT với một cấu hình baseline. Chưa có kết quả sweep GPU trong lần cập nhật này. Mục tiêu bổ sung là kiểm tra learning rate ảnh hưởng thế nào đến Macro-F1 validation và lỗi theo nhãn/phong cách.

Fine-tuning cập nhật trọng số pretrained trên dữ liệu của nhóm. Tuning hyperparameter thử nhiều cấu hình rồi chọn bằng validation. BiGRU hiện được huấn luyện từ đầu; không gọi đó là fine-tune pretrained.

## Thiết kế cố định trước khi chạy

| Thành phần | Thiết lập |
| --- | --- |
| Learning rate thử nghiệm | `1e-5`, `2e-5`, `3e-5` |
| Đối chứng | Chạy lại `2e-5` trong cùng runtime; giữ baseline đã có riêng |
| Mô hình | `vinai/phobert-base-v2`, revision trong `configs/midterm.json`; cập nhật toàn bộ backbone và head |
| Dữ liệu | Cùng train sau loại trùng và cùng validation; preprocessing giữ nguyên |
| Huấn luyện | Seed 42, batch 16, 4 epoch, max length 256 BPE, AdamW weight decay 0,01, BCEWithLogitsLoss |
| Chọn epoch trong mỗi lượt | Macro-F1 validation cao nhất tại threshold 0,5; hòa chọn epoch sớm |
| Chọn learning rate | Macro-F1 validation cao nhất tại threshold 0,5; hòa chọn learning rate nhỏ hơn |
| Test | Chỉ tham gia audit dữ liệu có sẵn; không suy luận, tính metric hoặc chọn cấu hình bằng test |

Mỗi lượt bắt đầu lại từ cùng pretrained revision và seed. Không nạp trọng số đã fine-tune của lượt trước. Không dùng điểm sau threshold sweep để chọn learning rate trong thí nghiệm này. Giữ số epoch cố định để kiểm tra learning rate trong cùng ngân sách huấn luyện; chưa thể kết luận đã hội tụ.

## Chạy trên Colab

1. Tại project local, tạo ZIP bằng `.venv/bin/python scripts/package_colab.py --include-results`. ZIP chứa code, tài liệu và kết quả tổng hợp hiện có để build lại report; không chứa CSV nguồn, checkpoint hay vocabulary.
2. Mở [notebook 05](../notebooks/05_colab_phobert_tuning.ipynb) trên Colab, bật T4 GPU và chạy các cell theo thứ tự. Upload ZIP code, rồi upload riêng ba CSV được cấp quyền.
3. Notebook kiểm tra baseline artifacts, setup VnCoreNLP, chạy cả ba learning rate và kiểm chứng kết quả. Trên lượt baseline cũ, PhoBERT mất khoảng 51 phút/4 epoch; thời gian ba lượt mới phụ thuộc runtime, chưa được đo.
4. Notebook build lại report/slide và tải ZIP kết quả. Giải nén ZIP vào project local để nhập kết quả. Checkpoint tốt nhất được tải riêng bằng cell cuối; lưu vào nơi riêng tư trước khi Colab hết hạn.
5. Đọc các ca lỗi trong dữ liệu nội bộ, điền các cột review và bổ sung diễn giải vào report/slide. Không công khai comment gốc.

Các lệnh notebook thực hiện, từ project root:

```bash
python scripts/tune_phobert.py run
python scripts/build_report.py
```

Có thể kiểm tra lại kết quả đã nhập mà không cần GPU:

```bash
.venv/bin/python scripts/tune_phobert.py summarize
.venv/bin/python scripts/build_report.py
```

Chuẩn bị cấu hình mà chưa huấn luyện: `python scripts/tune_phobert.py prepare`. Lệnh này yêu cầu thư mục đầu ra chưa tồn tại; bản cập nhật hiện có thể đã chuẩn bị sẵn thư mục mặc định. `run` nhận thư mục đã prepare nếu chưa có kết quả lượt chạy nào.

Nếu sweep bị gián đoạn sau khi một learning rate đã hoàn tất, dùng `run --resume` như dưới đây. Nếu chính một lượt chỉ có output/checkpoint dở dang, script dừng và giữ nguyên file; không hỗ trợ resume từ giữa epoch hoặc tự xóa để train lại. Khôi phục đầy đủ artifact từ bản sao lưu hoặc dùng một study riêng với tên mới. Report mặc định đọc `reports/phobert_tuning`; không trộn các lượt thuộc các sweep khác nhau.

## Tiếp tục sweep bị gián đoạn trên Colab

Giữ runtime và thư mục study hiện có. Upload **chỉ file [tune_phobert.py mới](../scripts/tune_phobert.py)**; không extract lại ZIP lên study, không chạy lại cell `prepare`, cài lại package hoặc xóa thư mục kết quả/checkpoint.

```python
%cd /content/ViGoEmotions
from google.colab import files
from pathlib import Path

uploaded = files.upload()  # Chỉ chọn tune_phobert.py đã cập nhật
Path('scripts/tune_phobert.py').write_bytes(uploaded['tune_phobert.py'])
```

Sau đó chạy:

```python
import subprocess, sys

subprocess.run([
    sys.executable, '-u', 'scripts/tune_phobert.py', 'run', '--resume',
    '--output', 'reports/phobert_tuning',
    '--vncorenlp-dir', 'data/vncorenlp',
], check=True)
```

Với `lr_1e-5` hoàn chỉnh và hai lượt còn lại chưa có output/checkpoint, log sẽ báo `SKIP lr_1e-5`, rồi `RUN lr_2e-5` và `RUN lr_3e-5`. Chỉ bỏ qua lượt có config khớp, đủ history epoch liên tiếp, metric khớp xác suất/epoch được chọn, archive validation hợp lệ và metadata môi trường đầy đủ. Không cần tải/nạp lại checkpoint để kiểm chứng đầu ra; checkpoint đã có được giữ nguyên.

Một thư mục rỗng hoặc chỉ có checkpoint vẫn là lượt dở dang, không phải lượt hoàn tất. Script kiểm tra mọi lượt trước khi train bất kỳ learning rate còn thiếu nào. Data checksum, Python/nền tảng, package versions và tên GPU hiện tại phải khớp các lượt đã hoàn tất; lỗi in rõ trường nào khác nhau và không chọn winner. Không bỏ qua kiểm tra này khi Colab cấp GPU/package khác.

Sau mỗi lượt mới, kiểm tra lại artifact và môi trường. Chỉ gọi summarize khi cả ba lượt hợp lệ, cùng ID/ground truth validation và môi trường tương thích. Không dự đoán, tính metric hoặc chọn cấu hình bằng test; metadata checksum nguồn vẫn dùng quy ước audit hiện có của project. Không có `--resume`, `run` vẫn từ chối output hoặc thư mục checkpoint đã tồn tại như trước.

Khi lệnh thành công, chạy tiếp các cell build report, tải ZIP kết quả và tải checkpoint được chọn trong notebook 05. Nếu runtime đã mất, cần khôi phục study/config/output từ backup và tạo môi trường tương thích trước khi resume; chỉ có checkpoint là chưa đủ khôi phục một lượt hoàn chỉnh.

## Kết quả được lưu ở đâu?

```text
reports/phobert_tuning/
  study.json                      thiết kế, trạng thái, lượt được chọn, checksum xác suất
  configs/lr_*.json                ba cấu hình chỉ khác learning rate
  lr_1e-5/                        từng lượt có run JSON, tables/, figures/, predictions/
  lr_2e-5/
  lr_3e-5/
  comparison.csv                  metric tại 0,5 và delta so với đối chứng cùng runtime
  per_label_comparison.csv        F1 từng nhãn và delta
  styles_comparison.csv           F1 theo phong cách và delta
  learning_curves.png             train BCE loss và validation Macro-F1 qua epoch
  neural_error_review.csv         ca lỗi của lượt được chọn, cột review ban đầu để trống
  experiment_summary.md           tóm tắt số liệu; không thay audit thủ công
data/checkpoints/phobert_tuning/  checkpoint riêng từng lượt, giữ private
```

Script kiểm tra cùng ID, ground truth và thứ tự 27 nhãn; cùng cấu hình ngoài learning rate; cùng checksum dữ liệu, môi trường và GPU. Metric được tính lại từ xác suất và đối chiếu checkpoint được chọn trước khi công bố sweep hoàn tất. Baseline gốc vẫn nằm ở `reports/tables/model_comparison.csv`.

## Viết insight sau khi có kết quả

- **Kết quả tổng thể:** learning rate được chọn là gì? Delta so với lượt `2e-5` trong cùng sweep bao nhiêu? Nếu đối chứng thắng, ghi rằng hai thay đổi chưa cải thiện điểm trong grid này.
- **Đường học:** train loss giảm ra sao, validation F1 còn tăng ở epoch cuối hay dao động? Hiện lưu train loss và validation F1; chưa có validation loss. Một đường cong không đủ chứng minh chắc chắn overfitting.
- **Nhãn và phong cách:** nhãn nào tăng/giảm? Teencode hoặc emoji-heavy có cùng xu hướng với toàn validation không? Các nhóm có thể chồng lấn.
- **Ca lỗi:** đọc 20 FN và 20 FP ứng viên của lượt được chọn, đối chiếu xác suất của đối chứng qua cùng sample ID. Điền `reviewer`, `observation`, `hypothesis`; phân biệt điều quan sát được và giả thuyết giải thích.
- **Giới hạn:** một seed, một split, ba learning rate, bốn epoch; validation đã dùng chọn cấu hình. Chưa kết luận ý nghĩa thống kê hoặc khả năng tổng quát hóa trên test.

Sau khi chốt neural model cho Final, giữ checkpoint cố định và nghiên cứu fixed/global/per-label threshold theo đề. Đổi threshold chỉ thay cách giải mã xác suất; không retrain backbone để so sánh ba decoder.
