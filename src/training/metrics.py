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


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Tính điểm BLEU-4 chuẩn (sacrebleu).

    Trả về điểm BLEU từ 0–100 (giống sacrebleu convention).
    Dự phòng bằng cách tính BLEU-1 đơn giản nếu sacrebleu chưa cài.
    """
    try:
        from sacrebleu.metrics import BLEU as SacreBLEU
        bleu = SacreBLEU(effective_order=True)
        result = bleu.corpus_score(predictions, [references])
        return float(result.score)
    except Exception:
        # Fallback: tính BLEU-1 đơn giản nếu sacrebleu không dùng được
        from collections import Counter
        import math

        def _ngrams(tokens, n):
            return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))

        total_clip, total_count, total_ref_len, total_pred_len = 0, 0, 0, 0
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.strip().split()
            ref_tokens = ref.strip().split()
            pred_ngrams = _ngrams(pred_tokens, 1)
            ref_ngrams = _ngrams(ref_tokens, 1)
            clipped = {ng: min(cnt, ref_ngrams[ng]) for ng, cnt in pred_ngrams.items()}
            total_clip += sum(clipped.values())
            total_count += max(len(pred_tokens), 1)
            total_ref_len += len(ref_tokens)
            total_pred_len += len(pred_tokens)

        precision = total_clip / max(total_count, 1)
        bp = math.exp(min(0, 1 - total_ref_len / max(total_pred_len, 1)))
        return round(bp * precision * 100, 2)


def compute_token_accuracy(predictions: List[str], references: List[str]) -> float:
    """Tính Token Accuracy (cấp âm tiết/từ): tỷ lệ từ được dự đoán đúng
    trên tổng số vị trí token trong câu tham chiếu.

    Đây là chỉ số 'Accuracy' phù hợp cho bài toán sửa lỗi chính tả:
    đánh giá ở mức độ từng từ, không yêu cầu toàn câu khớp nhau.
    """
    total_correct = 0
    total_tokens = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()
        length = max(len(pred_tokens), len(ref_tokens))
        pred_padded = pred_tokens + [""] * (length - len(pred_tokens))
        ref_padded = ref_tokens + [""] * (length - len(ref_tokens))
        total_correct += sum(1 for p, r in zip(pred_padded, ref_padded) if p == r)
        total_tokens += max(length, 1)
    return total_correct / max(total_tokens, 1)


def compute_detection_metrics(
    predictions: List[str],
    sources: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Tính Precision, Recall, F1 cho bài toán phát hiện lỗi chính tả
    ở cấp độ âm tiết (syllable-level detection).

    Định nghĩa:
      - True Positive  (TP): vị trí có lỗi thực (source ≠ reference) VÀ hệ thống đã sửa (pred ≠ source).
      - False Positive (FP): vị trí không có lỗi (source = reference) NHƯNG hệ thống đã sửa (pred ≠ source).
      - False Negative (FN): vị trí có lỗi thực (source ≠ reference) NHƯNG hệ thống bỏ sót (pred = source).

    Args:
        predictions: Danh sách câu đã sửa (output của model/engine).
        sources:     Danh sách câu gốc bị lỗi (input text).
        references:  Danh sách câu chuẩn (gold label / target text).

    Returns:
        Dict gồm precision (%), recall (%), f1 (%), true_positive, false_positive, false_negative.
    """
    tp = fp = fn = 0

    for pred, src, ref in zip(predictions, sources, references):
        pred_toks = pred.strip().split()
        src_toks  = src.strip().split()
        ref_toks  = ref.strip().split()

        length = max(len(src_toks), len(ref_toks), len(pred_toks))
        src_toks  += [""] * (length - len(src_toks))
        ref_toks  += [""] * (length - len(ref_toks))
        pred_toks += [""] * (length - len(pred_toks))

        for s, r, p in zip(src_toks, ref_toks, pred_toks):
            has_error   = (s != r)   # vị trí thực sự có lỗi (gold)
            was_changed = (p != s)   # hệ thống đã thay đổi vị trí này

            if has_error and was_changed:
                tp += 1
            elif not has_error and was_changed:
                fp += 1
            elif has_error and not was_changed:
                fn += 1

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "true_positive":  tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision":      round(precision * 100, 2),
        "recall":         round(recall * 100, 2),
        "f1":             round(f1 * 100, 2),
    }


def build_compute_metrics_fn(tokenizer) -> Callable:
    """Hàm tạo callback compute_metrics chuẩn cho HuggingFace Seq2SeqTrainer.

    Trong quá trình huấn luyện, tính EM, CER, WER và BLEU trên validation set
    sau mỗi epoch để theo dõi tiến trình học.
    """

    def compute_metrics(eval_preds) -> Dict[str, float]:
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Giải mã các token IDs về văn bản
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        em   = compute_exact_match(decoded_preds, decoded_labels)
        cer  = compute_cer(decoded_preds, decoded_labels)
        wer  = compute_wer(decoded_preds, decoded_labels)
        bleu = compute_bleu(decoded_preds, decoded_labels)

        return {
            "exact_match": round(em * 100, 2),
            "cer":         round(cer * 100, 2),
            "wer":         round(wer * 100, 2),
            "bleu":        round(bleu, 2),
        }

    return compute_metrics
