# Thực nghiệm trên phân chia VSEC cố định

Chạy baseline và các cấu hình Transformer từ gốc dự án.

```sh
python -X utf8 run_experiments.py exp01 exp02 exp03 exp04 exp05
python -X utf8 run_experiments.py exp07 --candidate experiments/exp02/best_model experiments/exp03/best_model experiments/exp04/best_model experiments/exp05/best_model
```

| Lần chạy | Mô hình | Dữ liệu | Epoch tối đa |
|---|---|---|---:|
| EXP01 | N-gram + Levenshtein | VSEC train | |
| EXP02 | ViT5 | VSEC train | 1 |
| EXP03 | ViT5 | VSEC train | 3 |
| EXP04 | ViT5 | VSEC train, câu sạch, 5.000 mẫu nhiễu | 3 |
| EXP05 | ViT5 | VSEC train, câu sạch, 10.000 mẫu nhiễu | 5 |
| EXP06 | ByT5 | VSEC train, câu sạch, 5.000 mẫu nhiễu | 3 |
| EXP07 | ViT5 | Chọn checkpoint và thiết lập giải mã bằng validation | |

Các cấu hình `noise01`, `noise03`, `noise20k`, `lr_high`, `lr_low`, `batch8` thử xác suất nhiễu, số mẫu, learning rate và batch size. EXP06 dành cho môi trường đủ GPU. Có thể chọn ByT5 bằng `python -X utf8 select_checkpoint.py --model byt5 --checkpoints experiments/exp06/best_model`.

Mỗi kết quả lưu SHA-256 của dữ liệu, các chỉ số, lịch sử loss và ví dụ lỗi. Các lần so sánh dùng cùng tập đánh giá và cùng phiên bản manifest. Dùng validation để chọn cấu hình trước khi đo test.

Để thử tỷ lệ gần 40% VSEC, 20% mất dấu toàn câu, 15% mất dấu một phần, 15% lỗi hỗn hợp và 10% câu đúng, chạy:

~~~sh
python -X utf8 train.py --output experiments/vit5-accent --noise-size 9340 --identity-size 1868 --epochs 3
python -X utf8 select_checkpoint.py --model vit5 --checkpoints experiments/vit5-accent/best_model
python -X utf8 evaluate.py --model vit5 --split test --stress
~~~

Tùy chọn --stress báo cáo riêng Detection và Correction theo nhóm lỗi, dùng các biến thể từ câu chuẩn của tập đánh giá. Các biến thể này chỉ dùng để đánh giá và được kiểm tra trùng với Train.
