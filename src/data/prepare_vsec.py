"""Script tải và tiền xử lý bộ dữ liệu chuẩn VSEC Corpus (Vietnamese Spelling Error Correction).
Nguồn: https://github.com/VSEC2021/VSEC
Bộ dữ liệu gồm 9.341 câu và 11.202 lỗi chính tả thực tế được gán nhãn thủ công.
"""

import json
import os
import random
import sys
import urllib.request
from pathlib import Path


VSEC_RAW_URL = "https://raw.githubusercontent.com/VSEC2021/VSEC/main/Dataset/VSEC.jsonl"
RAW_FILE = Path("data/raw/VSEC.jsonl")
PROCESSED_DIR = Path("data/processed")


def download_vsec():
    """Tải file VSEC.jsonl từ kho chính thức."""
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and RAW_FILE.stat().st_size > 20_000_000:
        print(f"[*] File {RAW_FILE} đã tồn tại ({RAW_FILE.stat().st_size / (1024*1024):.2f} MB). Bỏ qua tải.")
        return

    print(f"[*] Đang tải VSEC Corpus từ {VSEC_RAW_URL}...")
    req = urllib.request.Request(VSEC_RAW_URL, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req) as resp, open(RAW_FILE, "wb") as f:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = downloaded * 100 / total_size
                print(f"    -> Đã tải {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="\r")
            else:
                print(f"    -> Đã tải {downloaded / (1024*1024):.1f} MB", end="\r")
    print(f"\n[*] Tải thành công VSEC Corpus: {RAW_FILE}")


def process_vsec():
    """Chuyển đổi VSEC sang format train/val/test jsonl chuẩn."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    records = []
    clean_sentences = []
    
    print("[*] Đang đọc và chuyển đổi VSEC.jsonl...")
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            
            ann = item.get("annotations", [])
            if not ann:
                continue
            
            # Khôi phục input và target theo từng âm tiết
            input_tokens = []
            target_tokens = []
            error_count = 0
            
            for a in ann:
                curr = a.get("current_syllable", "")
                alt = a.get("alternative_syllables", [])
                is_correct = a.get("is_correct", True)
                
                input_tokens.append(curr)
                if not is_correct and alt:
                    target_tokens.append(alt[0])
                    error_count += 1
                else:
                    target_tokens.append(curr)
            
            input_text = " ".join(input_tokens).strip()
            target_text = " ".join(target_tokens).strip()
            
            if not input_text or not target_text:
                continue
            
            records.append({
                "id": f"vsec_{idx+1:05d}",
                "input_text": input_text,
                "target_text": target_text,
                "error_count": error_count,
            })
            clean_sentences.append(target_text)

    print(f"[*] Đã trích xuất được {len(records)} mẫu câu từ VSEC Corpus.")
    total_errors = sum(r["error_count"] for r in records)
    print(f"[*] Tổng số lỗi chính tả: {total_errors}")

    # Xáo trộn dữ liệu với seed cố định để tái lập
    random.seed(42)
    random.shuffle(records)

    # Chia tập train (80%), val (10%), test (10%)
    n = len(records)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_data = records[:n_train]
    val_data = records[n_train:n_train + n_val]
    test_data = records[n_train + n_val:]

    # Ghi file JSONL
    for name, data in [("train.jsonl", train_data), ("val.jsonl", val_data), ("test.jsonl", test_data)]:
        out_path = PROCESSED_DIR / name
        with open(out_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[*] Đã lưu {name}: {len(data)} mẫu câu ({out_path})")

    # Cập nhật thêm câu sạch vào data/raw/raw_corpus.txt để bổ trợ từ điển và N-gram
    raw_corpus_path = Path("data/raw/raw_corpus.txt")
    if clean_sentences:
        with open(raw_corpus_path, "w", encoding="utf-8") as f:
            for s in clean_sentences:
                f.write(s + "\n")
        print(f"[*] Đã cập nhật ngữ liệu sạch vào {raw_corpus_path} ({len(clean_sentences)} câu)")


if __name__ == "__main__":
    download_vsec()
    process_vsec()
