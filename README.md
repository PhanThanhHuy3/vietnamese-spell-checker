# VietSpell AI

Phát hiện vị trí sai chính tả trong văn bản tiếng Việt và gợi ý âm tiết thay thế theo ngữ cảnh. Website cho phép xem văn bản gốc, áp dụng từng thay đổi, hoàn tác và sửa tất cả.

## Chạy trên máy

Từ thư mục dự án, chạy bằng Python 3.10 trở lên.

```sh
python -m pip install -r requirements.txt
python -X utf8 -m src.data.prepare_vsec
python -X utf8 -m src.data.validate_split
python -X utf8 evaluate.py --model baseline
python -X utf8 app.py
```

Mở [VietSpell AI](http://127.0.0.1:7860/#check). Baseline N-gram + Levenshtein chạy trên CPU. Bộ khôi phục dấu tìm chuỗi âm tiết bằng beam search từ thống kê VSEC Train và ngữ liệu tiếng Việt Tatoeba. Hệ thống giữ phương án thay thế cho những âm tiết có nhiều cách thêm dấu. Trình kiểm tra chọn ViT5 khi có checkpoint triển khai đã xác minh. Có thể dùng cùng pipeline từ dòng lệnh.

```sh
python -X utf8 predict.py --text "Hôm nai tôi đến chường để học lập chình."
python -X utf8 -m pytest -q
```

Gợi ý của hệ thống có thể sai. Giao diện giữ bản gốc và đánh dấu vị trí gợi ý để người dùng xem trước. Sửa tự động áp dụng gợi ý qua ngưỡng chênh lệch điểm ngữ cảnh. Sửa tất cả áp dụng danh sách đã duyệt, trừ các mục được bỏ qua. Vị trí API dùng chỉ số ký tự Unicode từ 0, điểm kết thúc loại trừ. Hệ thống giữ xuống dòng, dấu câu và các đoạn được nhận diện là URL, email, tên riêng hoặc thuật ngữ. Nhận diện tên riêng dựa trên mẫu viết hoa và danh sách thuật ngữ.

## Dữ liệu và đánh giá

Manifest VSEC lưu 7.472 câu train, 934 câu validation và 935 câu test cùng SHA-256 và ID. Pipeline nhóm câu có cùng văn bản nguồn, đầu vào hoặc đáp án đã chuẩn hóa trước khi chia. Dữ liệu nhiễu, cặp câu sạch và thống kê N-gram học từ train. Test được dùng để đo cuối cùng. Xem [quy trình dữ liệu](data/README.md).

```sh
python -X utf8 evaluate.py --model baseline --split val
python -X utf8 evaluate.py --model baseline --split practice
python -X utf8 evaluate.py --model baseline --split test --stress
```

Kết quả được ghi vào `experiments/results/<split>/` và hiển thị trên website. Detection chấm vị trí. Correction yêu cầu đồng thời đúng vị trí và đúng nội dung sửa. Báo cáo cũng có Exact Match, Token Accuracy, BLEU-4, CER, WER và tỷ lệ can thiệp vào câu chuẩn đối chứng. Tập `practice` gồm các ví dụ soạn tay để kiểm tra hành vi. Đánh giá trên dữ liệu VSEC dùng các bản ghi thực tế của corpus.

## ViT5 và ByT5

ViT5 là mô hình Transformer encoder–decoder dùng token con. ByT5 mã hóa byte UTF-8. Cả hai học chuỗi đầu vào có tiền tố `spelling: ` và chuỗi mục tiêu sạch. Vòng huấn luyện dùng cross-entropy, AdamW, dynamic padding, chọn checkpoint theo validation loss và early stopping. Các cặp vượt giới hạn token được báo theo ID.

```sh
python -X utf8 train.py --output experiments/exp04 --epochs 3 --noise-size 5000 --identities
python -X utf8 select_checkpoint.py --model vit5 --checkpoints experiments/exp04/best_model
python -X utf8 evaluate.py --model vit5
```

`select_checkpoint.py` chọn beam và ngưỡng log-xác suất trên validation, đồng thời tính mức sửa quá tay trên câu chuẩn. Chạy trên GPU với notebook [Colab](notebooks/train_colab.ipynb) hoặc dùng [ma trận thí nghiệm](experiments/README.md). Trọng số cũ từng dùng phân chia có rò rỉ. Quy trình triển khai yêu cầu metadata của phân chia hiện tại. Một lần chạy smoke xác minh huấn luyện, lưu và nạp lại trọng số ViT5. Chỉ số của smoke và checkpoint cũ được loại khỏi báo cáo thực nghiệm trên web.

## Nguồn

- [Tatoeba, bản tải ngữ liệu tiếng Việt](https://tatoeba.org/en/downloads)
- [VSEC Corpus](https://github.com/VSEC2021/VSEC)
- [VSEC Paper](https://arxiv.org/abs/2111.00640)
- [ViT5 Paper](https://aclanthology.org/2022.naacl-srw.18/)
- [VietAI/vit5-base](https://huggingface.co/VietAI/vit5-base)
- [ByT5 Paper](https://arxiv.org/abs/2105.13626)
