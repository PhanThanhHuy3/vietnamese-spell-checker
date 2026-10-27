from typing import Callable, Dict, List

import numpy as np


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Tính tỷ lệ dự đoán chính xác hoàn toàn (Exact Match - EM)."""
    if not predictions or not references:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / len(predictions)


def compute_levenshtein(s1: str, s2: str) -> int:
    """Tính khoảng cách Levenshtein giữa 2 chuỗi ký tự."""
    if len(s1) < len(s2):
        return compute_levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """Tính Character Error Rate (CER) không phụ thuộc bắt buộc vào thư viện ngoài."""
    total_dist = 0
    total_chars = 0
    for p, r in zip(predictions, references):
        total_dist += compute_levenshtein(p, r)
        total_chars += max(len(r), 1)
    return total_dist / max(total_chars, 1)


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """Tính Word Error Rate (WER) ở cấp độ từ."""
    total_dist = 0
    total_words = 0
    for p, r in zip(predictions, references):
        words_p = p.strip().split()
        words_r = r.strip().split()
        total_dist += compute_levenshtein(" ".join(words_p), " ".join(words_r))
        total_words += max(len(words_r), 1)
    return total_dist / max(total_words, 1)


def build_compute_metrics_fn(tokenizer) -> Callable:
    """Hàm tạo callback compute_metrics chuẩn cho HuggingFace Seq2SeqTrainer."""

    def compute_metrics(eval_preds) -> Dict[str, float]:
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Giải mã các token IDs về văn bản
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        em = compute_exact_match(decoded_preds, decoded_labels)
        cer = compute_cer(decoded_preds, decoded_labels)
        wer = compute_wer(decoded_preds, decoded_labels)

        return {
            "exact_match": round(em * 100, 2),
            "cer": round(cer * 100, 2),
            "wer": round(wer * 100, 2),
        }

    return compute_metrics
