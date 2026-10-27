# 🇻🇳 Phát Hiện và Tự Động Sửa Lỗi Chính Tả Tiếng Việt
*(Vietnamese Spelling Detection and Correction using Deep Learning & Transformers)*

Dự án áp dụng mô hình **HuggingFace Transformers (ViT5 / PhoBERT)** và **PyTorch** để phát hiện và tự động sửa các loại lỗi chính tả tiếng Việt thường gặp (lỗi gõ telex, lỗi ngữ âm/phương ngữ ch/tr, s/x, d/gi/r, lỗi dấu hỏi/ngã, rụng dấu, v.v.).

---

## 📁 Cấu trúc Thư mục Chuẩn (Project Structure)

Dự án được tổ chức theo chuẩn cấu trúc module mã nguồn Python hiện đại:

```text
vietnamese-spell-checker/
├── configs/                          # Chứa các file cấu hình YAML/JSON
│   └── config.yaml                   # Cấu hình siêu tham số, dữ liệu, model
├── data/                             # Quản lý toàn bộ dữ liệu
│   ├── raw/                          # Văn bản thô (clean text corpus)
│   ├── synthetic/                    # Dữ liệu sinh nhiễu nhân tạo
│   ├── processed/                    # Dữ liệu train.jsonl, val.jsonl, test.jsonl
│   └── README.md                     # Hướng dẫn định dạng & quy chuẩn dữ liệu
├── notebooks/                        # Jupyter Notebooks thực nghiệm & phân tích
│   └── 01_quickstart_and_eda.ipynb   # Demo tạo nhiễu, tiền xử lý, tính metrics
├── checkpoints/                      # Lưu trữ trọng số model & tokenizer sau khi huấn luyện
├── logs/                             # Nhật ký huấn luyện và kết quả đánh giá
├── src/                              # Mã nguồn chính của dự án (Source code)
│   ├── __init__.py
│   ├── data/                         # Tiền xử lý dữ liệu & sinh lỗi chính tả
│   │   ├── __init__.py
│   │   ├── preprocess.py             # Chuẩn hóa Unicode NFC, dấu thanh, khoảng trắng
│   │   ├── noise_synthesizer.py      # Bộ sinh nhiễu lỗi chính tả tiếng Việt
│   │   └── dataset.py                # PyTorch Dataset & DataCollator Seq2Seq
│   ├── models/                       # Kiến trúc mô hình
│   │   ├── __init__.py
│   │   └── spell_corrector.py        # Model wrapper cho ViT5 / Transformer Seq2Seq
│   ├── training/                     # Vòng lặp huấn luyện & đánh giá
│   │   ├── __init__.py
│   │   ├── metrics.py                # Exact Match (EM), CER, WER
│   │   └── trainer.py                # Wrapper Seq2SeqTrainer của HuggingFace
│   ├── utils/                        # Các hàm tiện ích
│   │   ├── __init__.py
│   │   ├── helpers.py                # Đọc config, set seed, kiểm tra GPU
│   │   └── logger.py                 # Logging chuẩn hoá
│   └── inference.py                  # Pipeline suy luận & so khớp tìm vị trí lỗi
├── generate_noise_data.py            # Script sinh dữ liệu huấn luyện từ văn bản sạch
├── train.py                          # Script huấn luyện mô hình
├── evaluate.py                       # Script đánh giá mô hình trên tập test
├── predict.py                        # Script thử nghiệm sửa lỗi tương tác từ bàn phím
├── requirements.txt                  # Danh sách thư viện phụ thuộc
├── .gitignore                        # Cấu hình bỏ qua các file tạm/weights lớn
└── README.md                         # Tài liệu hướng dẫn sử dụng
```

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### 1. Tạo môi trường ảo và cài đặt thư viện
```bash
# Khởi tạo môi trường ảo Python (khuyên dùng Python 3.10 hoặc 3.11)
python -m venv venv

# Kích hoạt môi trường (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# (Hoặc trên Linux / MacOS: source venv/bin/activate)

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Chuẩn bị & Sinh dữ liệu lỗi nhân tạo (Data Synthesis)
Bài toán sửa lỗi chính tả thường khan hiếm dữ liệu câu sai thực tế. Bạn có thể đặt văn bản sạch vào `data/raw/raw_corpus.txt` hoặc chạy trực tiếp script để sinh dữ liệu mẫu:
```bash
python generate_noise_data.py --noise_prob 0.3
```
*Script sẽ tự động chuẩn hóa văn bản, áp dụng các quy tắc gây nhiễu và lưu thành `train.jsonl`, `val.jsonl`, `test.jsonl` trong thư mục `data/processed/`.*

### 3. Huấn luyện mô hình (Training)
Chạy huấn luyện mô hình ViT5 với cấu hình trong `configs/config.yaml`:
```bash
python train.py --epochs 5 --batch_size 16 --lr 5e-5
```
Trọng số tốt nhất (Best Model) sẽ tự động được lưu tại: `checkpoints/vit5_spelling_corrector/best_model/`.

### 4. Đánh giá mô hình (Evaluation)
Đánh giá độ chính xác của mô hình trên tập kiểm thử với các độ đo **Exact Match (EM)**, **Character Error Rate (CER)**, và **Word Error Rate (WER)**:
```bash
python evaluate.py --model_path checkpoints/vit5_spelling_corrector/best_model
```

### 5. Khởi chạy Giao diện Web trực quan (Web UI)
Bạn có thể mở giao diện Web tương tác trực quan ngay lập tức:
```bash
python app.py
```
*Trình duyệt sẽ tự động mở tại `http://localhost:7860` với 2 tab:*
- **Tab 1 - Bộ Sinh Nhiễu (Noise Generator):** Điều chỉnh tỷ lệ lỗi, chọn bật/tắt các quy luật (Telex/VNI, QWERTY, Vùng miền, Bỏ dấu) và xem chi tiết lỗi được tô màu sinh động.
- **Tab 2 - Sửa Lỗi Chính Tả (AI Corrector):** Nhập câu cần sửa, hệ thống sẽ phát hiện từ sai và hiển thị bảng so sánh Diff trực quan.

### 6. Thử nghiệm sửa lỗi qua dòng lệnh (CLI Terminal)
- **Sửa một câu cụ thể:**
  ```bash
  python predict.py --text "Hôm nai chời mưa to qúa, tôi quến mang ô."
  ```
- **Chế độ tương tác nhập trực tiếp trong terminal:**
  ```bash
  python predict.py
  ```
- **Thử nghiệm nhanh script sinh nhiễu:**
  ```bash
  python src/noise_generator.py
  ```

---

## 🧠 Lựa chọn Mô hình (ViT5 vs PhoBERT)

1. **ViT5 (`VietAI/vit5-base` hoặc `VietAI/vit5-large`) - Khuyên dùng (Seq2Seq)**:
   - Bản chất là mô hình sinh chuỗi (Encoder-Decoder T5).
   - Xử lý mượt mà các lỗi phức tạp: 1 từ sai thành nhiều từ, nhiều từ gộp thành 1 từ, đảo từ, bỏ sót từ.
   - Là lựa chọn chuẩn hàng đầu hiện nay cho Spelling Correction tiếng Việt.

2. **PhoBERT (`vinai/phobert-base-v2`) (Encoder-only / MLM / Token Classification)**:
   - Thích hợp cho bài toán **Phát hiện vị trí lỗi (Error Detection)** ở cấp độ token (gán nhãn nhị phân 0/1: từ đúng hay từ sai).
   - Có thể dùng kết hợp trong mô hình hai giai đoạn: *PhoBERT phát hiện lỗi -> ViT5 sửa lỗi*.

---

## ⚙️ Cấu hình chi tiết (`configs/config.yaml`)
Bạn có thể dễ dàng điều chỉnh các tham số trong `configs/config.yaml`:
- `model.name_or_path`: Mô hình nền tảng từ HuggingFace Hub.
- `model.max_source_length` & `max_target_length`: Độ dài tối đa của câu đầu vào và đầu ra.
- `training.learning_rate`, `batch_size`, `fp16`: Các siêu tham số tối ưu hóa phần cứng.
- `generation.num_beams`: Số lượng chùm tia khi giải mã văn bản.
