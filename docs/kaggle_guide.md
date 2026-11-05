# Chạy ViGoEmotions trên Kaggle

**Luồng chính mới:** [hai notebook 08/09 riêng cho 27/28 nhãn, cùng 20 epoch](kaggle_equal_epochs.md). Hướng dẫn 04/05 bên dưới giữ cho các lượt lịch sử.

Các lượt chạy dùng [profile cấu hình tách riêng](reproducibility_profiles.md). Notebook 04 mặc định baseline 10/4; profile Kaggle 25/20 giữ riêng, không sửa đè `midterm.json`. Notebook/gói chia sẻ đã xóa output.

## 1. File đã chuẩn bị

- `notebooks/04_kaggle_neural_baselines.ipynb`: chạy EDA, TF-IDF, BiGRU, PhoBERT và tạo báo cáo.
- `notebooks/05_kaggle_phobert_tuning.ipynb`: chạy ba learning rate PhoBERT từ pretrained model, dùng baseline đã đóng gói để tạo báo cáo.
- `vigoemotions_colab.zip`: gói code dùng chung cho cả Colab và Kaggle, có kết quả baseline cũ; không chứa CSV nguồn/checkpoint/vocabulary.

Hai notebook là bản chuyển phần nhập/xuất của notebook Colab; mô hình và cấu hình thí nghiệm giữ nguyên. Chưa chạy GPU trên Kaggle để xác minh end-to-end.

## 2. Tạo hai Dataset riêng tư

Trên Kaggle chọn Create > New Dataset (tên nút có thể thay đổi).

1. Tạo `vigoemotions-code`, upload `vigoemotions_colab.zip`. Giữ Private rồi tạo Dataset.
2. Tạo `vigoemotions-data`, upload ba file gốc `train.csv`, `val.csv`, `test.csv`. Giữ Private rồi tạo Dataset.

Đây là cách gắn file vào runtime, không phải bài toán thi đấu hay thao tác submit. Dữ liệu của project có giới hạn sử dụng; giữ cả Dataset, notebook và output riêng tư.

## 3. Import notebook

Vào Code > New Notebook, dùng File > Import Notebook hoặc nút Upload Notebook để chọn `04_kaggle_neural_baselines.ipynb`. Kiểm tra notebook là Private.

Trong panel Input/Data, bấm Add Input hoặc Add Data, tìm trong Your Datasets và attach cả hai Dataset vừa tạo. Code tự tìm file bên trong `/kaggle/input`, không cần sửa tên slug của Dataset. Chỉ attach một bộ ba CSV để tránh tên trùng.

## 4. Chọn môi trường

Trong Settings/Session options chọn Accelerator là GPU khả dụng và bật Internet. GPU phục vụ huấn luyện neural; Internet phục vụ cài dependency và tải PhoBERT/VnCoreNLP. Không chọn TPU vì code dùng PyTorch CUDA. Nếu có hai GPU, code hiện tại chỉ dùng GPU thứ nhất.

## 5. Chạy từng cell

1. **Khởi tạo project:** copy/giải nén code từ Input sang `/kaggle/working/ViGoEmotions`. Input là nơi đọc file; Working là nơi chạy và ghi kết quả. Cell dừng nếu project đã tồn tại để tránh ghi đè. Khi chạy tiếp trong cùng session, bỏ qua cell này.
2. **Cài thư viện:** dùng `requirements-colab.txt`, kiểm tra và cài Java nếu thiếu. Tên file có chữ colab nhưng thư viện không chỉ dành cho Colab. Không dùng `requirements.txt` khóa theo môi trường local.
3. **Kiểm tra GPU/Java:** phải thấy tên GPU và Java version. Nếu GPU assertion lỗi, kiểm tra Accelerator trước khi chạy tiếp.
4. **Nạp dữ liệu:** copy ba CSV về `data/`, kiểm tra header `id,text,labels`. Test vẫn cần cho quy trình kiểm tra dữ liệu/checksum, không dùng chọn mô hình.
5. **Helper:** định nghĩa hàm ghi log và hiển thị kết quả.
6. **Kiểm thử, EDA, TF-IDF:** phải hoàn tất trước neural; các bước này kiểm tra pipeline dữ liệu và tạo baseline CPU.
7. **BiGRU:** huấn luyện 10 epoch theo cấu hình, lưu checkpoint có validation Macro-F1 cao nhất tại ngưỡng 0.5.
8. **PhoBERT:** tải VnCoreNLP và pretrained model, huấn luyện 4 epoch, chọn checkpoint tương tự.
9. **Báo cáo:** tạo bảng so sánh, đường học, report/slide từ kết quả mới.
10. **Xuất file:** chạy cả cell ZIP kết quả và cell ZIP checkpoint riêng tư. Đường dẫn file được in ra; không còn hộp download của Colab.

Dừng ở cell lỗi, đọc thông báo/log rồi xử lý. Không chạy cell báo cáo khi training thất bại: gói code chứa kết quả cũ nên có thể tạo bảng lẫn lượt cũ và mới. Thời gian trên Kaggle chưa đo; không suy ra thời gian từ số đo Colab.

## 6. Lưu và tải kết quả

Hai file ZIP nằm trực tiếp trong `/kaggle/working`:

- `vigoemotions_results.zip`: bảng, hình, log, xác suất, report/slide và split IDs.
- `vigoemotions_baseline_private.zip`: checkpoint hai mô hình, vocabulary GRU và metadata cần giữ riêng tư.

Tải cả hai bằng file browser/Output. Dùng Save Version để giữ phiên bản notebook và output; đọc lựa chọn trước khi lưu: Save & Run All sẽ chạy lại toàn bộ notebook trong lượt mới, tốn thêm thời gian GPU. Chỉ coi output đã được giữ khi version hoàn tất thành công và bạn thấy file trong Output. Tải bản sao trước khi kết thúc session.

## 7. Tuning PhoBERT

Tạo notebook khác bằng `05_kaggle_phobert_tuning.ipynb`, attach cùng hai Dataset, bật GPU + Internet rồi chạy từ trên xuống. Notebook này không cần chạy lại BiGRU hay notebook 04 trước vì gói code đã có baseline để đối chiếu/báo cáo.

Ba learning rate `1e-5`, `2e-5`, `3e-5` đều bắt đầu từ pretrained revision gốc, không bắt đầu từ checkpoint tốt nhất của notebook 04. Giữ các thông số khác cố định và chạy cùng môi trường để so sánh. Thư mục study mặc định được đóng gói ở trạng thái prepared; không chạy thêm lệnh prepare.

Tải `vigoemotions_tuning_results.zip` và checkpoint được chọn (đường dẫn in ở cell cuối). Checkpoint nằm bên trong project ở `/kaggle/working/ViGoEmotions/data/checkpoints/phobert_tuning/`.

Không dùng `--resume` để nối sweep Colab vào Kaggle khi khác GPU, Python hoặc package versions: script kiểm tra và sẽ từ chối. Nếu study đã có output thì không giải nén code đè lên; xem `docs/phobert_tuning.md` về resume và giới hạn không hỗ trợ tiếp tục giữa epoch.

## 8. Các lỗi thường gặp

| Thông báo | Cách xử lý |
| --- | --- |
| Không tìm thấy gói code hoặc CSV | Attach đủ hai Dataset; kiểm tra Input chứa đúng tên file |
| Tìm thấy nhiều train.csv | Bỏ Dataset thừa để chỉ còn một bộ CSV |
| Project đã tồn tại | Khi tiếp tục session, bỏ qua cell khởi tạo; muốn chạy mới thì dùng session/workspace mới |
| GPU assertion lỗi | Bật GPU trong Session options và chạy lại cell kiểm tra |
| Lỗi tải pretrained/VnCoreNLP | Kiểm tra Internet và lỗi mạng cụ thể trong output |
| Lỗi dependency sau pip | Restart kernel nếu Kaggle yêu cầu; chạy lại từ cell kiểm tra GPU, bỏ qua khởi tạo nếu thư mục vẫn tồn tại |
| CUDA out of memory | Giảm batch_size trong configs/midterm.json trước training và ghi nhận thay đổi; tuning phải dùng cùng batch cho cả ba lượt |
| Output/checkpoint đã tồn tại khi tuning | Không xóa bừa; giữ bản sao, dùng study/session mới hoặc quy trình resume được hỗ trợ |

## 9. Cập nhật gói code

Sau khi sửa project local, chạy `python scripts/prepare_kaggle.py` để tái tạo hai notebook và ZIP. Tạo version mới cho Dataset code trên Kaggle và kiểm tra notebook đang dùng đúng version.

Tài liệu Kaggle: https://www.kaggle.com/docs/notebooks và https://www.kaggle.com/docs/datasets.

## 10. Notebook 06 — thí nghiệm 28 nhãn gồm Neutral

Chạy `python3 scripts/prepare_kaggle.py --neutral28` để tạo riêng notebook 06 và ZIP code mới.
Import `notebooks/06_kaggle_neutral_28_labels.ipynb`; attach Dataset Private chứa
`vigoemotions_neutral28_colab.zip` cùng Dataset ba CSV, bật GPU + Internet rồi chạy từ trên xuống.
Không dùng ZIP của notebook 04/05. Epoch lấy từ config trong ZIP, được in sau init.

Project chạy tại `/kaggle/working/ViGoEmotions28`; ZIP cuối là
`/kaggle/working/vigoemotions_neutral28_private_results.zip`, gồm cả kết quả và checkpoint riêng tư.
Hướng dẫn đầy đủ và cách chạy lại workspace: [neutral_28_labels.md](neutral_28_labels.md#chạy-kaggle-gpu--notebook-06).
