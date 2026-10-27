import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.models.spell_corrector import VietnameseSpellCorrector
from src.training.metrics import compute_cer, compute_exact_match, compute_wer
from src.utils.helpers import get_device, load_config
from src.utils.logger import setup_logger

logger = setup_logger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình sửa lỗi chính tả trên tập Test.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Đường dẫn file cấu hình")
    parser.add_argument("--model_path", type=str, default=None, help="Đường dẫn tới checkpoint mô hình cần đánh giá")
    parser.add_argument("--test_file", type=str, default=None, help="Đường dẫn tới file test.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = args.model_path or str(Path(config["training"]["output_dir"]) / "best_model")
    test_file = Path(args.test_file or config["data"]["test_file"])

    if not Path(model_path).exists():
        logger.warning(f"Chưa tìm thấy checkpoint tại '{model_path}'. Dùng baseline model '{config['model']['name_or_path']}'.")
        model_path = config["model"]["name_or_path"]

    if not test_file.exists():
        logger.error(f"Không tìm thấy file kiểm thử tại: {test_file}")
        return

    device = get_device()
    logger.info(f"Đang tải mô hình từ: {model_path} trên {device}...")
    corrector = VietnameseSpellCorrector(model_name_or_path=model_path, device=device)

    # Đọc dữ liệu test
    inputs, targets = [], []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                inputs.append(item["input_text"])
                targets.append(item["target_text"])

    logger.info(f"Số lượng câu kiểm thử: {len(inputs)}. Đang tiến hành dự đoán...")
    predictions = []
    batch_size = 16

    for i in tqdm(range(0, len(inputs), batch_size), desc="Đánh giá"):
        batch_inputs = inputs[i : i + batch_size]
        batch_preds = corrector.correct(batch_inputs, max_length=config["model"]["max_target_length"])
        predictions.extend(batch_preds)

    # Tính toán độ đo
    em = compute_exact_match(predictions, targets)
    cer = compute_cer(predictions, targets)
    wer = compute_wer(predictions, targets)

    print("\n" + "=" * 50)
    print("           KẾT QUẢ ĐÁNH GIÁ (EVALUATION REPORT)     ")
    print("=" * 50)
    print(f"Tổng số mẫu kiểm thử: {len(inputs)}")
    print(f"Exact Match (EM)   : {em * 100:.2f}%")
    print(f"Character Error Rate: {cer * 100:.2f}%")
    print(f"Word Error Rate (WER): {wer * 100:.2f}%")
    print("=" * 50)

    # Hiển thị vài ví dụ so sánh
    print("\nVí dụ thực tế:")
    for idx in range(min(5, len(inputs))):
        print(f"[{idx + 1}]")
        print(f" - Input   : {inputs[idx]}")
        print(f" - Target  : {targets[idx]}")
        print(f" - Predict : {predictions[idx]}")
        print("-" * 40)


if __name__ == "__main__":
    main()
