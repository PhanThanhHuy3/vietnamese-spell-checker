# 🇻🇳 Phát Hiện và Tự Động Sửa Lỗi Chính Tả Tiếng Việt
*(Vietnamese Automated Spelling Correction and Error Detection using Deep Learning & Transformers)*

Dự án áp dụng mô hình **Transformer Seq2Seq (ViT5)** và **PyTorch** để phát hiện và tự động sửa các loại lỗi chính tả tiếng Việt thường gặp: lỗi gõ Telex/VNI, lỗi ngữ âm/phương ngữ (ch/tr, s/x, d/gi/r), lỗi dấu hỏi/ngã, rụng dấu, nhầm phím QWERTY, v.v.

Hệ thống sử dụng **VSEC Corpus** — bộ dữ liệu chuẩn tiếng Việt với 9.341 câu và 11.202 lỗi chính tả thực tế được gán nhãn thủ công.

---

## 🧠 Kiến Trúc Mô Hình

### Mô hình Deep Learning Chính: ViT5 (Transformer Seq2Seq)

```
Input câu lỗi → [Encoder ViT5] → [Decoder ViT5] → Output câu đã sửa
                    T5-base            Beam Search
                  (SentencePiece       num_beams=4
                   tokenization)
```

- **Kiến trúc**: T5 Encoder-Decoder (Transformer), ~250M parameters
- **Pretrained backbone**: `VietAI/vit5-base` (pretrained trên 100GB văn bản tiếng Việt)
- **Finetuning**: Cross-Entropy Loss, AdamW optimizer, Warmup Scheduler
- **Decoding**: Beam Search (`num_beams=4`, `no_repeat_ngram_size=4`, `repetition_penalty=1.2`)

### Mô hình Bổ Sung: Smart Corrector (N-gram + Levenshtein)

- Từ điển Viet74K (underthesea) + pyvi + VSEC corpus
- Bigram Language Model (bigram scored candidates)
- Khoảng cách Levenshtein để tìm từ sửa đổi gần nhất
- Bộ giải mã Telex, VNI, Teencode, phương ngữ Bắc-Trung-Nam

---

## 📁 Cấu Trúc Thư Mục

```text
vietnamese-spell-checker/
├── configs/
│   └── config.yaml                   # Hyperparameters (lr, batch, epochs, beam…)
├── data/
│   ├── raw/
│   │   ├── VSEC.jsonl                # VSEC Corpus thô (25.1 MB, 9341 câu)
│   │   └── raw_corpus.txt            # Câu sạch trích từ VSEC
│   ├── processed/
│   │   ├── train.jsonl               # 7.472 cặp (input_lỗi, target_chuẩn)
│   │   ├── val.jsonl                 # 934 cặp
│   │   ├── test.jsonl                # 935 cặp
│   │   ├── lm_unigram.json           # Unigram LM
│   │   └── lm_bigram.json            # Bigram LM (1.3 MB)
│   └── README.md
├── notebooks/
│   ├── 01_quickstart_and_eda.ipynb   # EDA & demo sinh nhiễu
│   └── train_colab.ipynb             # Notebook huấn luyện Google Colab (T4 GPU)
├── checkpoints/
│   └── vit5_spelling_corrector/
│       └── best_model/               # Checkpoint ViT5 đã finetuned
├── logs/
│   ├── EVALUATION.md                 # Bảng so sánh kết quả các model
│   └── eval_results.json             # Kết quả đánh giá dạng JSON
├── src/
│   ├── data/
│   │   ├── prepare_vsec.py           # Tải & parse VSEC Corpus từ GitHub
│   │   ├── preprocess.py             # Normalize Unicode NFC, dấu thanh
│   │   ├── noise_synthesizer.py      # Sinh lỗi nhân tạo (6 chiến lược)
│   │   └── dataset.py                # PyTorch Dataset & DataCollator Seq2Seq
│   ├── models/
│   │   └── spell_corrector.py        # ViT5 wrapper (load/save/correct)
│   ├── training/
│   │   ├── metrics.py                # EM, BLEU, CER, WER, Token Acc, Detection P/R/F1
│   │   └── trainer.py                # HuggingFace Seq2SeqTrainer wrapper
│   ├── utils/
│   │   ├── helpers.py                # load_config, set_seed, get_device
│   │   └── logger.py                 # Logging chuẩn
│   ├── smart_corrector.py            # Rule engine N-gram (không cần GPU)
│   └── inference.py                  # Pipeline inference + phát hiện vị trí lỗi
├── generate_noise_data.py            # Sinh dữ liệu huấn luyện từ văn bản sạch
├── train.py                          # Huấn luyện / finetuning ViT5
├── evaluate.py                       # Đánh giá toàn diện (8 chỉ số định lượng)
├── predict.py                        # Dự đoán CLI tương tác
├── app.py                            # Gradio Web UI (port 7860)
└── requirements.txt
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Môi trường ảo & cài thư viện

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell
# source venv/bin/activate           # Linux/macOS

pip install -r requirements.txt
```

### 2. Tải & chuẩn bị VSEC Corpus

```bash
# Tải VSEC Corpus (~25 MB) và chia train/val/test
python src/data/prepare_vsec.py
```

### 3. Sinh dữ liệu nhiễu nhân tạo (Data Augmentation)

```bash
python generate_noise_data.py --noise_prob 0.3
```

Script sinh 6 loại nhiễu: Telex, phương ngữ, dấu hỏi/ngã, rụng dấu, QWERTY, lỗi cơ học.

### 4. Huấn luyện mô hình ViT5 (Training / Finetuning)

```bash
python train.py --epochs 5 --batch_size 16 --lr 5e-5
```

Checkpoint tốt nhất được lưu tại: `checkpoints/vit5_spelling_corrector/best_model/`

Trên Google Colab (T4 GPU): mở `notebooks/train_colab.ipynb`

### 5. Đánh giá mô hình (Evaluation — 8 chỉ số định lượng)

```bash
# Đánh giá model đã finetuned
python evaluate.py --model_path checkpoints/vit5_spelling_corrector/best_model

# Đánh giá với tên định danh tùy chọn (ghi vào bảng so sánh)
python evaluate.py \
    --model_path checkpoints/vit5_spelling_corrector/best_model \
    --run_name "vit5_finetuned_ep5" \
    --num_examples 10
```

**Các chỉ số được tính:**

| Chỉ số | Loại | Mô tả |
|---|---|---|
| **Exact Match (EM)** | Correction | % câu dự đoán hoàn toàn chính xác |
| **Token Accuracy** | Correction | % từ được dự đoán đúng (cấp âm tiết) |
| **BLEU-4** | Correction | Điểm BLEU-4 (sacrebleu, 0–100) |
| **CER** | Correction | Character Error Rate (↓ tốt hơn) |
| **WER** | Correction | Word Error Rate (↓ tốt hơn) |
| **Precision** | Detection | Tỷ lệ sửa đúng / tổng vị trí đã sửa |
| **Recall** | Detection | Tỷ lệ phát hiện đúng / tổng lỗi thực |
| **F1-score** | Detection | Trung bình điều hòa Precision & Recall |

### 6. Giao diện Web (Gradio UI)

```bash
python app.py
# Mở trình duyệt tại http://localhost:7860
```

- **Tab 1**: Bộ Sinh Nhiễu — tô màu từng loại lỗi được tạo
- **Tab 2**: AI Corrector — hiển thị Diff so sánh input vs output

### 7. Dự đoán qua dòng lệnh

```bash
# Sửa một câu
python predict.py --text "Hôm nai chời mưa to qúa, tôi quến mang ô."

# Chế độ tương tác
python predict.py
```

---

## 📊 Kết Quả Thực Nghiệm

Xem bảng so sánh đầy đủ tại [`logs/EVALUATION.md`](logs/EVALUATION.md).

| Model | #Câu | EM (%) | BLEU | Token Acc (%) | CER (%) | WER (%) | Det-P (%) | Det-R (%) | F1 (%) |
|---|---|---|---|---|---|---|---|---|---|
| Smart Corrector (rule) | 200 | 36.50 | 92.24 | — | 1.05 | 3.59 | 59.42 | 71.00 | 64.69 |
| ViT5 Finetuned | 935 | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |

> Chạy `python evaluate.py` để cập nhật kết quả ViT5.

---

## ⚙️ Cấu Hình (`configs/config.yaml`)

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `model.name_or_path` | `VietAI/vit5-base` | Pretrained backbone |
| `model.max_source_length` | 64 | Độ dài tối đa input |
| `training.learning_rate` | 2e-4 | AdamW learning rate |
| `training.num_train_epochs` | 2 | Số epoch huấn luyện |
| `training.per_device_train_batch_size` | 12 | Batch size |
| `training.warmup_ratio` | 0.1 | Warmup ratio |
| `training.fp16` | false | Mixed precision (true khi có GPU) |
| `generation.num_beams` | 4 | Beam Search width |

---

## 📚 Tài Liệu Tham Khảo

- **VSEC Corpus**: [github.com/VSEC2021/VSEC](https://github.com/VSEC2021/VSEC) — 9.341 câu, 11.202 lỗi thực tế
- **ViT5**: [VietAI/vit5-base](https://huggingface.co/VietAI/vit5-base) — T5 pretrained tiếng Việt
- **PhoBERT**: [vinai/phobert-base-v2](https://huggingface.co/vinai/phobert-base-v2) — BERT tiếng Việt
- Raffel et al. (2020) — *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*
