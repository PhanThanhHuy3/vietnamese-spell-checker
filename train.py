import argparse
import os
from pathlib import Path

# Ưu tiên sử dụng cache local để không bị nghẽn mạng
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.data.dataset import SpellingDataset
from src.models.spell_corrector import VietnameseSpellCorrector
from src.training.trainer import SpellingTrainer
from src.utils.helpers import get_device, load_config, set_seed
from src.utils.logger import setup_logger

logger = setup_logger("train")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình sửa lỗi chính tả Tiếng Việt.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size trên mỗi GPU/CPU")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    args = parser.parse_args()

    # 1. Đọc cấu hình
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["per_device_train_batch_size"] = args.batch_size
        config["training"]["per_device_eval_batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    set_seed(config.get("seed", 42))
    device = get_device()
    logger.info(f"Khởi động tiến trình huấn luyện trên thiết bị: {device}")

    # 2. Khởi tạo mô hình và Tokenizer
    model_name = config["model"]["name_or_path"]
    logger.info(f"Đang tải pretrained model: {model_name}...")
    model_wrapper = VietnameseSpellCorrector(model_name_or_path=model_name, device=device)

    # 3. Chuẩn bị dữ liệu
    train_file = Path(config["data"]["train_file"])
    val_file = Path(config["data"]["val_file"])

    if not train_file.exists() or not val_file.exists():
        logger.error(f"Không tìm thấy file dữ liệu huấn luyện. Hãy chạy 'python generate_noise_data.py' trước!")
        return

    logger.info("Đang nạp tập dữ liệu Train và Validation...")
    train_dataset = SpellingDataset(
        file_path=train_file,
        tokenizer=model_wrapper.tokenizer,
        max_source_length=config["model"]["max_source_length"],
        max_target_length=config["model"]["max_target_length"],
    )
    val_dataset = SpellingDataset(
        file_path=val_file,
        tokenizer=model_wrapper.tokenizer,
        max_source_length=config["model"]["max_source_length"],
        max_target_length=config["model"]["max_target_length"],
    )
    logger.info(f"Số lượng mẫu Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # 4. Huấn luyện
    trainer = SpellingTrainer(
        model_wrapper=model_wrapper,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=config,
    )

    logger.info("Bắt đầu huấn luyện...")
    trainer.train()
    logger.info("Huấn luyện hoàn tất! Model tốt nhất đã được lưu vào thư mục đầu ra.")


if __name__ == "__main__":
    main()
