# Các lần thực nghiệm VietSpell AI

| Lần chạy | Mô hình | Dữ liệu và mục đích |
|---|---|---|
| EXP00 | Giữ nguyên input | Mức nền của các chỉ số |
| EXP01 | N-gram + Levenshtein | Baseline truyền thống |
| EXP02 | ViT5-base | VSEC Train |
| EXP03 | ViT5-base | VSEC Train và mẫu lỗi nhân tạo |
| EXP04 | ViT5-base | VSEC Train, mẫu lỗi nhân tạo và câu sạch |
| EXP05 | ViT5-base | Kiểm tra learning rate và số epoch khác |
| EXP06 | ViT5-base | Chọn checkpoint và cấu hình hậu xử lý trên Validation |
| EXP07 | ByT5-small | Mô hình đối chứng tùy chọn |

Các lệnh được điều phối bởi `run_experiments.py`. EXP06 nhận danh sách ứng viên qua `--candidate`. Sau khi chọn cấu hình, dùng `evaluate.py --model vit5 --split test --stress` để đo trên Test. Báo cáo Tăng cường dữ liệu dùng cùng bộ sinh lỗi. Dữ liệu demo người dùng nhập chỉ trả về hoặc tải xuống từ trình duyệt.

`evaluate.py` lưu chỉ số và từng dự đoán trong `experiments/results/<split>/`. Chọn `--split clean_test` để đo tỷ lệ bảo toàn câu đúng và `--split challenge` để xem các ví dụ kiểm tra ngoài VSEC. Accuracy cú pháp âm tiết, phát hiện vị trí, thay đúng từ và Exact Match là các phép đo riêng.

Nhật ký training và manifest hoàn chỉnh được ghi khi fine-tuning kết thúc. Checkpoint được chọn chỉ xuất hiện trong dropdown sau khi khớp manifest và chạy thử thành công.
