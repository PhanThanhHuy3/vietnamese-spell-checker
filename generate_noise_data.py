import argparse
import json
from pathlib import Path
from typing import List

from src.data.noise_synthesizer import VietnameseNoiseSynthesizer
from src.data.preprocess import normalize_vietnamese_text
from src.utils.helpers import load_config, set_seed
from src.utils.logger import setup_logger

logger = setup_logger("generate_noise_data")


def split_data(samples: List[dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Chia danh sách mẫu thành train, val, test."""
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = samples[:train_end]
    val_data = samples[train_end:val_end]
    test_data = samples[val_end:]
    return train_data, val_data, test_data


def save_jsonl(data: List[dict], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Đã lưu {len(data)} mẫu vào {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Sinh tập dữ liệu nhiễu nhân tạo cho bài toán sửa lỗi chính tả.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Đường dẫn file cấu hình")
    parser.add_argument("--input_file", type=str, default=None, help="File text chứa các câu tiếng Việt sạch")
    parser.add_argument("--noise_prob", type=float, default=None, help="Tỷ lệ gây nhiễu mỗi từ")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    input_file = Path(args.input_file or config["data"].get("raw_file", "data/raw/raw_sample.txt"))
    noise_prob = args.noise_prob or config["data"].get("noise_prob", 0.3)

    if not input_file.exists():
        logger.warning(f"Không tìm thấy file {input_file}. Đang tạo dữ liệu mẫu sạch...")
        input_file.parent.mkdir(parents=True, exist_ok=True)
        sample_sentences = [
            "Hôm nay trời rất đẹp và trong xanh.",
            "Trường đại học Bách Khoa Hà Nội có bề dày lịch sử.",
            "Trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc mỗi ngày.",
            "Học phần học sâu cung cấp kiến thức nền tảng về mạng nơ-ron.",
            "Tôi thường đọc sách và nghe nhạc vào buổi tối cuối tuần.",
            "Mô hình ngôn ngữ lớn có khả năng hiểu và sinh văn bản tự nhiên.",
            "Dự án này tập trung vào phát hiện và sửa lỗi chính tả Tiếng Việt.",
            "Thời tiết miền Bắc đang bước vào những ngày đầu đông se lạnh.",
            "Chúng ta cần kiên trì rèn luyện kỹ năng lập trình hàng ngày.",
            "ViT5 là mô hình transformer mạnh mẽ được huấn luyện cho tiếng Việt.",
            "Thư viện PyTorch rất linh hoạt và được cộng đồng nghiên cứu ưa chuộng.",
            "Lỗi chính tả thường do bộ gõ telex hoặc nhầm lẫn giữa các âm địa phương.",
        ] * 10
        with open(input_file, "w", encoding="utf-8") as f:
            for s in sample_sentences:
                f.write(s + "\n")
        logger.info(f"Đã tạo file mẫu tại {input_file} với {len(sample_sentences)} câu.")

    # Đọc dữ liệu sạch
    synthesizer = VietnameseNoiseSynthesizer(seed=config.get("seed", 42))
    dataset_pairs = []

    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            clean_text = normalize_vietnamese_text(line)
            if not clean_text:
                continue

            # Sinh câu bị lỗi
            noisy_text = synthesizer.add_noise(clean_text, noise_prob=noise_prob)
            dataset_pairs.append({
                "id": f"sample_{idx:05d}",
                "input_text": noisy_text,
                "target_text": clean_text,
            })

    logger.info(f"Tổng số mẫu đã tạo: {len(dataset_pairs)}")

    train_data, val_data, test_data = split_data(dataset_pairs)
    save_jsonl(train_data, Path(config["data"]["train_file"]))
    save_jsonl(val_data, Path(config["data"]["val_file"]))
    save_jsonl(test_data, Path(config["data"]["test_file"]))
    logger.info("Hoàn tất sinh dữ liệu.")


if __name__ == "__main__":
    main()
