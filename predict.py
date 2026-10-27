import argparse
from pathlib import Path
from src.inference import SpellInferencePipeline
from src.utils.helpers import load_config


def main():
    parser = argparse.ArgumentParser(description="Chương trình thử nghiệm sửa lỗi chính tả Tiếng Việt.")
    parser.add_argument("--text", type=str, default=None, help="Câu văn bản cần sửa lỗi")
    parser.add_argument("--model_path", type=str, default=None, help="Đường dẫn mô hình checkpoint")
    args = parser.parse_args()

    config = load_config("configs/config.yaml")
    model_path = args.model_path

    if not model_path:
        checkpoint_best = Path(config["training"]["output_dir"]) / "best_model"
        if checkpoint_best.exists():
            model_path = str(checkpoint_best)
        else:
            model_path = config["model"]["name_or_path"]

    print(f"[*] Đang tải mô hình từ: {model_path} ...")
    pipeline = SpellInferencePipeline(model_dir_or_name=model_path)
    print("[*] Sẵn sàng!\n")

    if args.text:
        res = pipeline.predict(args.text)
        print(f"Câu gốc: {res['original_text']}")
        print(f"Câu sửa: {res['corrected_text']}")
        if res["has_error"]:
            print(f"Phát hiện lỗi ({len(res['errors_found'])} vị trí):")
            for err in res["errors_found"]:
                print(f"  -> '{err['original_segment']}' ==> '{err['corrected_segment']}'")
        else:
            print("Không phát hiện lỗi chính tả.")
        return

    # Chế độ tương tác nhập trực tiếp từ bàn phím
    print("=" * 60)
    print("  CHẾ ĐỘ TƯƠNG TÁC SỬA LỖI CHÍNH TẢ TIẾNG VIỆT (Gõ 'exit' để thoát)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nNhập câu cần kiểm tra: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Tạm biệt!")
                break

            result = pipeline.predict(user_input)
            print("-" * 50)
            print(f"Kết quả sửa: {result['corrected_text']}")
            if result["has_error"]:
                print("Chi tiết chỉnh sửa:")
                for err in result["errors_found"]:
                    print(f"  - '{err['original_segment']}' => '{err['corrected_segment']}'")
            else:
                print("✓ Câu chuẩn, không cần chỉnh sửa.")
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát chương trình.")
            break


if __name__ == "__main__":
    main()
