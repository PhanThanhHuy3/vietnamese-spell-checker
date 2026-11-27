# VietSpell AI

Đồ án phát hiện vị trí sai chính tả tiếng Việt và đề xuất cách sửa theo ngữ cảnh. ViT5-base là Transformer chính, ByT5-small là mô hình đối chứng tùy chọn, N-gram + Levenshtein là baseline.

## Chạy ứng dụng

```sh
python -m pip install -r requirements.txt
python -X utf8 -m src.data.prepare_vsec
python -X utf8 -m src.data.validate_split
python -X utf8 app.py
```

Mở [VietSpell Editor](http://127.0.0.1:7860/editor) để kiểm tra chính tả. Hệ thống chọn mô hình khả dụng đã được xác minh. Bấm từ gạch chân để xem gợi ý và áp dụng ngay trong văn bản. **Sửa tự động** trong menu Công cụ áp dụng các gợi ý đủ điều kiện. **Sửa tất cả** áp dụng các gợi ý còn chờ. Kết quả có thể sao chép hoặc xuất TXT.

Mở [VietSpell AI Lab](http://127.0.0.1:7860/lab) để xem dữ liệu, so sánh mô hình, sinh lỗi, kiểm tra checkpoint và xem các báo cáo thực nghiệm.

Frontend nhận vị trí ký tự Unicode dưới dạng khoảng `[start, end)`. Phần chỉnh sửa giữ dấu câu, xuống dòng, URL, email và các vùng được bảo vệ theo quy tắc trong bộ xử lý. Trang thông tin mô hình hiển thị tên mô hình thực sự chạy.

## Checkpoint và dữ liệu

VSEC có 9.341 câu và 11.202 lỗi gán nhãn. Manifest lưu ID, SHA-256 và các cặp gần trùng đã được gom trong cùng tập. Website có thể đọc annotation gốc đã xác minh, hiển thị token ViT5, kiểm tra trùng lặp và xuất số liệu. N-gram dùng VSEC Train. Ngữ liệu tiếng Việt bổ sung dùng cho xếp hạng ngữ cảnh được lọc theo khóa nội dung của VSEC và các bộ câu thử. [VSEC Corpus](https://github.com/VSEC2021/VSEC)

Checkpoint cũ ở `checkpoints/vit5_spelling_corrector/best_model` có trọng số, config và tokenizer. Trang mô hình cho phép kiểm tra nạp riêng. Checkpoint triển khai mới cần metadata chứng minh phân chia hiện tại, kết quả chọn cấu hình trên Validation và một lượt inference thử thành công.

## Huấn luyện và đánh giá

Nếu có thư mục ViT5 pretrained cục bộ ở `vit5-base/`, cấu hình ViT5 kiểm tra SHA-256 của trọng số và mô hình SentencePiece trước khi nạp ngoại tuyến. Mục xem tokenization của VSEC cũng ưu tiên tokenizer cục bộ. Khi thư mục này vắng mặt, quy trình huấn luyện dùng định danh `VietAI/vit5-base`. Checkpoint sửa chính tả dùng phân chia hiện tại vẫn được xác minh riêng.

```sh
python -X utf8 run_experiments.py exp00 exp01
python -X utf8 run_experiments.py exp02 exp03 exp04
python -X utf8 run_experiments.py exp06 --candidate experiments/exp02/best_model experiments/exp03/best_model experiments/exp04/best_model
python -X utf8 evaluate.py --model vit5 --split test --stress --run_name vit5_final
python -X utf8 -m pytest -q
```

Fine-tuning dùng PyTorch, Cross-Entropy và AdamW. Validation loss dùng để giữ checkpoint ứng viên trong từng lần chạy. Bước triển khai chọn theo Validation Correction F1 và dùng tỷ lệ sửa câu sạch để phân xử khi bằng điểm. Thực nghiệm cuối dùng VSEC Test. Clean Sentence Test là tập riêng từ câu chuẩn của VSEC Test sau kiểm tra trùng với Train. Các biến thể mất dấu và lỗi hỗn hợp dùng chung nguồn VSEC Test với seed 42. Chi tiết ở [ma trận thực nghiệm](experiments/README.md) và [notebook Colab](notebooks/train_colab.ipynb).

Website đọc kết quả đã lưu. Kết quả Deep Learning có thêm dự đoán trước hậu xử lý. Số liệu của checkpoint cũ hoặc các lượt smoke không xuất hiện trong bảng kết quả chính thức.
