# Yêu cầu nộp ATI — cập nhật 19/11/2026

Nguồn đóng gói/định dạng hiện hành: [ATI_Submission_Guide.pdf](ATI_Submission_Guide.pdf), 5 trang, và [ATI_Final_Report_Template.docx](ATI_Final_Report_Template.docx). Các quy định cụ thể dưới đây thay các giả định trước đây về PDF report, slides Final và hai gói Midterm/Final. [Đề ViGoEmotions](02-vigoemotions-vietnamese-emotion.pdf) vẫn là nguồn yêu cầu thí nghiệm/RQ1.

## 1. Nộp một gói chung trước oral

- Một sinh viên đại diện nộp cho cả nhóm qua link Google Form/Drive giảng viên gửi.
- Không nộp riêng Midterm. Gói cuối chứa slides Midterm, code và báo cáo toàn project.
- Nộp trước ngày oral, theo hạn giảng viên thông báo; chưa có ngày chính thức trong hai tài liệu mới. Mốc 27/11 trong kế hoạch cũ chỉ là mục tiêu nội bộ.
- Nếu thay bản trước hạn, giữ đúng tên ZIP cũ; hướng dẫn ghi chấm bản cuối trước hạn.

## 2. Đặt tên

```text
<CLASS>_G<NN>_P02.zip
```

ViGoEmotions là **P02**. Mã lớp viết hoa, giữ dấu `-`; số nhóm và project có hai chữ số. Các mã ATI được guide liệt kê là `62FIT4ATI-CLC02` và `62FIT4ATI-CLC03`. Chưa được cung cấp mã lớp/số nhóm thực tế, nên không tự dùng G05 hay CLC02 từ ví dụ.

`<PREFIX>` là tên ZIP bỏ `.zip`. Không có dấu cách, tiếng Việt có dấu, tên thành viên hoặc chữ MIDTERM/FINAL trong tên ZIP.

Ví dụ **minh họa**, không phải mã nhóm đã xác nhận: `62FIT4ATI-CLC02_G05_P02.zip`.

## 3. Cấu trúc ZIP bắt buộc

```text
<PREFIX>.zip
└── <PREFIX>/
    ├── slides/
    │   └── <PREFIX>_midterm_slides.pdf
    ├── code/
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── notebooks/
    │   ├── src/                  (module dùng chung, tùy chọn)
    │   ├── configs/
    │   ├── splits/
    │   └── results/
    │       ├── tables/
    │       ├── figures/
    │       ├── predictions/
    │       └── environment.md
    └── report/
        └── <PREFIX>_report.docx
```

Có đúng một thư mục gốc trùng tên ZIP bỏ đuôi, với ba thư mục `slides/`, `code/`, `report/`. Không dùng hai folder nộp `midterm/` và `final/` như đề xuất cũ.

Guide mục 2 nói mọi file bên trong bắt đầu bằng PREFIX, nhưng cây ví dụ mục 3 vẫn dùng `code/README.md`, `requirements.txt` và tên code thông thường. Khi chuẩn bị gói, tuân tên ZIP/root/slides/report nêu rõ và giữ các tên code trong cây ví dụ; nếu cần áp PREFIX cho mọi source file thì phải làm rõ điểm diễn đạt này, không tự đổi tên làm hỏng imports/paths.

## 4. Slides và oral

- Chỉ nộp **slides Midterm**, tiếng Anh, PDF; không nộp PPTX.
- Dùng đúng bản đã trình chiếu. Không tự coi deck mới nhất trong Git là bằng chứng đã trình chiếu.
- Trang đầu cần tên học phần, mã lớp, nhóm, project, họ tên/MSSV mọi thành viên.
- **Không có slides Final.** Nội dung Final nằm trong report và được hỏi khi oral. Tài liệu ôn/Q&A vẫn có thể chuẩn bị riêng.

Tình trạng kiểm tra 19/11: deck ở `develop` có 15 trang; nhánh `docs/midterm-18-slides` có deck 18 trang. Chưa xác nhận bản đã trình chiếu thực tế. Trang đầu được đọc ở cả hai bản chưa có đủ thông tin nhóm/lớp/MSSV.

## 5. Report Word theo template

- Nộp **DOCX**, không nộp PDF thay thế. PDF có thể giữ nội bộ để đọc/kiểm bố cục.
- Dùng `ATI_Final_Report_Template.docx`, toàn bộ report bằng tiếng Anh.
- Bao gồm toàn project: Midterm là baseline/đối chứng, Final RQ1 là phần chính.
- Phần thân **Sections 1–7**, tới hết Conclusion, dài **15–20 trang**. Không tính cover, abstract, TOC, references hoặc appendices. Kiểm trên bố cục Word cuối cùng, không lấy số trang PDF cũ làm proof.
- A4, lề 2.5 cm; body Times New Roman 12 pt, giãn dòng 1.5, justified, đen trắng. Giữ font/size/color theo styles của template.
- Dùng Heading 1/2, Normal và Caption đúng vai trò. Caption bảng phía trên, caption hình phía dưới, đánh số tự động và được nhắc trong nội dung. Equations dùng Word Equation và có số bên phải.
- Abstract 150–250 từ, một đoạn, không citations; 3–5 keywords.
- IEEE citations theo thứ tự xuất hiện; trích dataset, pretrained models, model families, libraries và nguồn ý tưởng.
- Thay `[PREFIX]` ở running header. Xóa trang Template Instructions, hướng dẫn in nghiêng và thay placeholders; giữ các citations hợp lệ như `[1]`. Cập nhật toàn bộ TOC trước khi lưu.

### Cấu trúc phần thân

| Mục | Nội dung | Độ dài template gợi ý |
| --- | --- | --- |
| 1 | Introduction: motivation, RQ/hypothesis/control, contributions | Khoảng 2 trang |
| 2 | Background and Related Work | Khoảng 2.5 trang |
| 3 | Dataset and Task Framing, imbalance, leakage/split | Khoảng 2.5 trang |
| 4 | Methods and Experimental Protocol, metrics, compute | Khoảng 4 trang |
| 5 | Results: baseline recap, RQ1, trả lời giả thuyết | Khoảng 4 trang |
| 6 | Error Analysis, per-label errors, 3–5 failure examples | Khoảng 2 trang |
| 7 | Limitations, Ethics and Conclusion | Khoảng 1.5 trang |

Không tự thêm thí nghiệm để điền template. Bỏ dòng RQ2/subsection extension nếu chưa làm. Dòng caption mẫu “mean ± std over [n] seeds”, Python 3.11 và seeds `[42,43,44]` là ví dụ cần thay bằng runtime thật và **một seed 42** của study này; không bịa std hoặc thêm lần test. Thời gian inference chưa được ghi thì nêu rõ thay vì suy thành benchmark mới.

### Phụ lục template

- **Appendix A — Reproducibility:** lệnh cài/chạy, mapping từng bảng/hình về notebook/script; khớp `code/README.md`.
- **Appendix B — Member Contributions:** đủ mọi thành viên, kể cả đóng góp 0; % số nguyên, tổng 100; đánh dấu người đại diện `*`; cả nhóm thống nhất.
- **Appendix C — Use of Generative AI Tools:** tool, mục đích và sections/files bị tác động. Template yêu cầu mục này; phần lược bỏ AI wording trong bản report cũ chưa đáp ứng yêu cầu kê khai mới.

## 6. Code, outputs và môi trường

- Notebook chạy lại từ đầu trên Colab T4 miễn phí hoặc máy cá nhân; nộp **bản đã chạy, giữ outputs**.
- `code/README.md`: setup, link dữ liệu/điều khoản, thứ tự notebooks, mapping bảng/hình, link GitHub repo.
- Giữ fixed split IDs, configs/hyperparameters của mọi thí nghiệm được báo cáo.
- Predictions/probabilities cho **mọi model và điều kiện** được báo cáo, gồm các baseline và các decoder Final. Guide minh họa CSV; brief cho NPZ hoặc CSV. Có thể xuất CSV text-free từ saved NPZ để khớp cây gói, không inference lại.
- `code/results/environment.md`: GPU/hardware, Python, package versions, seeds và khác biệt môi trường nếu có.
- Dữ liệu/weights lớn để link Drive có quyền xem cho giảng viên, khi cần; README ghi cách lấy authorized inputs.

Không đưa vào ZIP: virtualenv, caches, `.git`, OS/IDE files, `.env`, tokens/passwords, raw restricted datasets hoặc checkpoints `.pt/.pth/.ckpt/.safetensors/.h5`. **Dưới 50 MB là khuyến nghị**, không được mô tả thành hard limit nếu chưa có thông báo khác.

## 7. Khoảng cách hiện tại và thứ tự xử lý

| Hạng mục | Tình trạng đã kiểm tra | Việc tiếp theo |
| --- | --- | --- |
| Report | Đã tạo [DOCX theo template](../reports/final_report_ati.docx), tích hợp baseline, Final RQ1 và RQ4 ở mục 5.3, thân 18 trang trong bản render kiểm tra, đủ Appendix A/B/C | Điền [metadata và kê khai](ati_report_fill_in.md), cập nhật TOC và kiểm lại phân trang trong Word sau khi điền |
| Metadata | Chưa có lớp/nhóm/tên/MSSV/đóng góp được cung cấp | Lấy thông tin thật, không suy từ Git author hoặc ví dụ G05 |
| Slides | Có hai deck 15/18 trang ở hai nhánh | Xác nhận bản đã trình chiếu và cách đáp ứng cover metadata |
| Midterm notebooks | Executed 08/09 có outputs; 01/02/07 cũng có executed outputs | Giữ outputs cần thiết, kiểm privacy và mapping report |
| Final notebooks | Đã nhập kết quả RQ1 và RQ4; notebook 10 và 11 trong repo là bản chạy mẫu | Export notebook đã chạy từ các lượt Kaggle, giữ outputs cho gói nộp |
| Predictions | Các NPZ Midterm/Final đã có | Gom đủ mọi model/condition vào layout nộp, xuất CSV từ saved scores nếu cần |
| ZIP builder | `submission_package.py` hiện xóa outputs; `package_final.py` tạo code ZIP chạy Kaggle | Làm packaging riêng cho submission, không dùng hai công cụ này nguyên trạng |
| Repro paths | Frozen artifacts/code bind các paths hiện tại | Mapping layout nộp phải bảo toàn source/gates và chạy được; không sửa hashes để ép gói mới qua kiểm tra |

Thứ tự tiếp theo: chốt metadata/slide nguồn → viết report theo template → hoàn thiện executed notebook và README/environment → đóng gói đúng PREFIX → kiểm unzip/layout/privacy/repro và TOC/pages. Chưa tạo gói nộp, chưa chạy thêm model và chưa thay thresholds trong lần đọc yêu cầu này.
