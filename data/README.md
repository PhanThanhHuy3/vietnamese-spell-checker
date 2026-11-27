# Dữ liệu VSEC

`python -X utf8 -m src.data.prepare_vsec` đọc tệp VSEC JSONL chính thức, khôi phục cặp câu từ annotation, chuẩn hóa NFC và khoảng trắng, giữ dấu câu, nhóm nội dung trùng và các cặp gần trùng đã rà soát, rồi cố định các tập với seed 42. Annotation có đề xuất chuỗi rỗng tạo thao tác xóa. Annotation rỗng tạo bởi khoảng trắng kép được bỏ qua khi ghép câu. Annotation có nhiều đề xuất dùng đề xuất đầu làm tham chiếu chính.

`data/splits/split_manifest.json` lưu ID, SHA-256, thống kê và các cặp gần trùng đã được đưa về cùng tập. Bộ rà soát dùng khóa 5 âm tiết đầu hoặc cuối và ngưỡng 90% tương đồng ký tự. Chạy `python -X utf8 -m src.data.validate_split` để xác minh ba tập được sinh đúng từ manifest. Khóa so trùng bỏ khác biệt hoa thường, dấu câu và khoảng trắng, đồng thời giữ dấu tiếng Việt.

`python -X utf8 generate_noise_data.py --size 5000 --noise_prob 0.2` tạo dữ liệu bổ sung trong `data/synthetic/train.jsonl`. Mỗi dòng chứa `parent_id` từ train. Pipeline tạo ba nhóm gồm mất dấu toàn câu, mất dấu một phần và lỗi hỗn hợp. Câu chuẩn có thể thêm bằng tùy chọn --identity-size. Pipeline loại các biến thể va chạm với nội dung validation hoặc test, kể cả khi bỏ dấu để kiểm tra. `train.py` dùng cùng hàm tăng cường với các tham số của lần thực nghiệm.

`data/practice.jsonl` là các mẫu soạn tay để kiểm tra hành vi theo đặc tả.

Nguồn [VSEC Corpus](https://github.com/VSEC2021/VSEC).

## Ngữ liệu cho bộ khôi phục dấu

Bản xuất chi tiết tiếng Việt của Tatoeba được lưu trong data/context để có thể chạy ngoại tuyến. Dữ liệu cung cấp thống kê ngữ cảnh cho bộ giải mã N-gram. Luồng fine-tuning ViT5 tiếp tục dùng VSEC Train và các biến thể sinh từ đó. Trình nạp loại nội dung trùng với bất kỳ bản ghi VSEC hay mẫu thực hành nào sau khi chuẩn hóa và bỏ dấu. Xem [nguồn và thông tin ghi công](context/README.md).

Clean Sentence Test được tạo từ target VSEC Test sau khi loại trùng với Train và lưu tại data/evaluation/clean_test.jsonl. Ngữ liệu VSEC không có ID tài liệu nguồn để kiểm tra hai câu đến từ cùng tài liệu.
