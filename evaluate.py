"""Script đánh giá toàn diện mô hình sửa lỗi chính tả tiếng Việt.

Các chỉ số định lượng được tính:
  - Accuracy   : Token Accuracy (cấp âm tiết)
  - Precision  : Precision phát hiện lỗi (syllable-level)
  - Recall     : Recall phát hiện lỗi (syllable-level)
  - F1-score   : F1 phát hiện lỗi (syllable-level)
  - BLEU       : BLEU-4 (sacrebleu)
  - Exact Match: Tỷ lệ câu dự đoán hoàn toàn chính xác
  - CER        : Character Error Rate
  - WER        : Word Error Rate
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.models.spell_corrector import VietnameseSpellCorrector
from src.training.metrics import (
    compute_bleu,
    compute_cer,
    compute_detection_metrics,
    compute_exact_match,
    compute_token_accuracy,
    compute_wer,
)
from src.utils.helpers import get_device, load_config
from src.utils.logger import setup_logger

logger = setup_logger("evaluate")

EVAL_RESULTS_JSON = Path("logs/eval_results.json")
EVAL_RESULTS_MD   = Path("logs/EVALUATION.md")


def _load_existing_results() -> dict:
    if EVAL_RESULTS_JSON.exists():
        try:
            with open(EVAL_RESULTS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"runs": {}}


def _save_results(all_results: dict) -> None:
    all_results["updated_at"] = datetime.now().isoformat(timespec="seconds")
    EVAL_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


def _update_markdown(all_results: dict) -> None:
    """Ghi lại bảng Markdown so sánh tất cả model đã evaluate."""
    lines = [
        "# Bảng Kết Quả Đánh Giá (Evaluation Report)\n",
        f"Cập nhật lần cuối: {all_results.get('updated_at', '—')}\n",
        "",
        "Tập test: VSEC test split (không dùng để chọn model).",
        "Chỉ số phát hiện lỗi tính ở cấp âm tiết.",
        "",
        "| Model | #Câu | EM (%) | BLEU | Accuracy (%) | CER (%) | WER (%) | Det-P (%) | Det-R (%) | Det-F1 (%) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for run_name, run_data in all_results.get("runs", {}).items():
        m = run_data.get("metrics", {})
        det = m.get("detection", {})
        lines.append(
            f"| {run_name} "
            f"| {m.get('num_samples', '—')} "
            f"| {m.get('exact_match', 0) * 100:.2f} "
            f"| {m.get('bleu', 0):.2f} "
            f"| {m.get('token_accuracy', 0) * 100:.2f} "
            f"| {m.get('cer', 0) * 100:.2f} "
            f"| {m.get('wer', 0) * 100:.2f} "
            f"| {det.get('precision', 0):.2f} "
            f"| {det.get('recall', 0):.2f} "
            f"| {det.get('f1', 0):.2f} |"
        )

    lines.append("")
    lines.append("> Bảng được sinh tự động bởi `python evaluate.py`.")

    EVAL_RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá toàn diện mô hình sửa lỗi chính tả tiếng Việt."
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Đường dẫn file cấu hình"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Đường dẫn tới checkpoint mô hình cần đánh giá"
    )
    parser.add_argument(
        "--test_file", type=str, default=None,
        help="Đường dẫn tới file test.jsonl"
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="Tên định danh cho lần evaluate này (ghi vào bảng so sánh)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size khi chạy inference (mặc định: 16)"
    )
    parser.add_argument(
        "--num_examples", type=int, default=5,
        help="Số ví dụ thực tế hiển thị sau khi đánh giá (mặc định: 5)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model_path or str(Path(config["training"]["output_dir"]) / "best_model")
    test_file  = Path(args.test_file or config["data"]["test_file"])

    if not Path(model_path).exists():
        logger.warning(
            f"Chưa tìm thấy checkpoint tại '{model_path}'. "
            f"Dùng baseline model '{config['model']['name_or_path']}'."
        )
        model_path = config["model"]["name_or_path"]

    if not test_file.exists():
        logger.error(f"Không tìm thấy file kiểm thử tại: {test_file}")
        return

    # --- 1. Nạp model ---
    device = get_device()
    logger.info(f"Đang tải mô hình từ: {model_path} trên {device}...")
    corrector = VietnameseSpellCorrector(model_name_or_path=model_path, device=device)

    # --- 2. Đọc dữ liệu test ---
    inputs, targets = [], []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                inputs.append(item["input_text"])
                targets.append(item["target_text"])

    logger.info(f"Số lượng câu kiểm thử: {len(inputs)}. Đang tiến hành dự đoán...")

    # --- 3. Chạy inference theo batch ---
    predictions = []
    max_len = config["model"]["max_target_length"]

    for i in tqdm(range(0, len(inputs), args.batch_size), desc="Đánh giá"):
        batch_inputs = inputs[i: i + args.batch_size]
        batch_preds  = corrector.correct(batch_inputs, max_length=max_len)
        predictions.extend(batch_preds)

    # --- 4. Tính toán tất cả chỉ số định lượng ---
    em             = compute_exact_match(predictions, targets)
    cer            = compute_cer(predictions, targets)
    wer            = compute_wer(predictions, targets)
    bleu           = compute_bleu(predictions, targets)
    token_acc      = compute_token_accuracy(predictions, targets)
    detection      = compute_detection_metrics(predictions, inputs, targets)

    # --- 5. Hiển thị kết quả ---
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"      KẾT QUẢ ĐÁNH GIÁ — {Path(model_path).name}")
    print(sep)
    print(f"  Số câu kiểm thử    : {len(inputs)}")
    print(f"  Model              : {model_path}")
    print(sep)
    print("  [Chỉ số Chỉnh sửa (Correction Metrics)]")
    print(f"  Exact Match (EM)   : {em * 100:.2f}%")
    print(f"  Token Accuracy     : {token_acc * 100:.2f}%")
    print(f"  BLEU-4             : {bleu:.2f}")
    print(f"  CER (↓ tốt hơn)   : {cer * 100:.2f}%")
    print(f"  WER (↓ tốt hơn)   : {wer * 100:.2f}%")
    print(sep)
    print("  [Chỉ số Phát Hiện Lỗi (Detection Metrics — syllable-level)]")
    print(f"  Precision          : {detection['precision']:.2f}%")
    print(f"  Recall             : {detection['recall']:.2f}%")
    print(f"  F1-score           : {detection['f1']:.2f}%")
    print(f"  TP / FP / FN       : {detection['true_positive']} / {detection['false_positive']} / {detection['false_negative']}")
    print(sep)

    # --- 6. Hiển thị ví dụ thực tế ---
    if args.num_examples > 0:
        print(f"\n{'─' * 65}")
        print(f"  VÍ DỤ THỰC TẾ (hiển thị {min(args.num_examples, len(inputs))} câu)")
        print(f"{'─' * 65}")
        for idx in range(min(args.num_examples, len(inputs))):
            print(f"  [{idx + 1}]")
            print(f"   Input  : {inputs[idx]}")
            print(f"   Target : {targets[idx]}")
            print(f"   Predict: {predictions[idx]}")
            is_correct = predictions[idx].strip() == targets[idx].strip()
            print(f"   Status : {'✓ ĐÚNG' if is_correct else '✗ SAI'}")
            print(f"  {'─' * 60}")

    # --- 7. Lưu kết quả vào logs/ ---
    run_label = args.run_name or Path(model_path).name
    all_results = _load_existing_results()
    all_results["runs"][run_label] = {
        "model_path":   model_path,
        "test_file":    str(test_file),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            "num_samples":   len(inputs),
            "exact_match":   round(em, 6),
            "token_accuracy": round(token_acc, 6),
            "bleu":          round(bleu, 4),
            "cer":           round(cer, 6),
            "wer":           round(wer, 6),
            "detection":     detection,
        },
    }
    _save_results(all_results)
    _update_markdown(all_results)

    logger.info(f"Đã lưu kết quả vào {EVAL_RESULTS_JSON} và {EVAL_RESULTS_MD}")


if __name__ == "__main__":
    main()
