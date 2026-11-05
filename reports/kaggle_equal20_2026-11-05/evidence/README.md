# Evidence Kaggle 20 epoch — kết quả chính midterm

Hai ZIP kết quả và notebook 08/09 executed là nguồn. `provenance.json` ở thư mục cha lưu SHA-256, đối chiếu LF/CRLF, môi trường và hash các thành viên được trích. Thư mục này chỉ chứa JSON và NPZ với ID/nhãn/xác suất, không có raw text, vocabulary hoặc checkpoint.

Mọi model được chấm lại trên validation. Tham số/thời gian lấy nguyên từ run JSON; bốn history neural đủ 20 epoch và checkpoint selection dùng emotion27 Macro-F1 tại 0,5. Checkpoint được xác minh có trong ZIP riêng tư; không nạp lại hay train local.

`27/` và `28/` giữ manifest/config/status/comparison nguyên bản của từng ZIP và từng model. Bảng comparison gốc có đường dẫn Kaggle lịch sử; bảng chính được tính lại ở thư mục cha. Các phân tích trong `../analysis/28_emotion27/` chỉ chấm 27 cột cảm xúc của model 28, không phải metric đủ 28 nhãn. Test chưa chấm.
