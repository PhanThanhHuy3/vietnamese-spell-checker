import argparse
from pathlib import Path
from src.smart_corrector import SmartVietnameseCorrector


def main():
    parser = argparse.ArgumentParser(description="Chương trình thử nghiệm sửa lỗi chính tả Tiếng Việt.")
    parser.add_argument("--text", type=str, default=None, help="Câu văn bản cần sửa lỗi")
    parser.add_argument("--model_path", type=str, default=None, help="Đường dẫn mô hình checkpoint (tùy chọn)")
    args = parser.parse_args()

    print("[*] Đang khởi động bộ kiểm tra chính tả tiếng Việt...")
    corrector = SmartVietnameseCorrector(model_path=args.model_path)
    engine_name = "Deep Learning (ViT5)" if corrector.use_ai else "Smart Heuristic & Phonetic Engine"
    print(f"[*] Chế độ hoạt động: {engine_name}")
    print("[*] Sẵn sàng!\n")

    if args.text:
        res = corrector.check_and_correct(args.text)
        print("=" * 60)
        print(f"Câu gốc   : {res['original_text']}")
        print(f"Câu sửa   : {res['corrected_text']}")
        if res["has_error"]:
            print(f"Phát hiện lỗi ({len(res['errors_found'])} vị trí):")
            for err in res["errors_found"]:
                print(f"  -> '{err['original_segment']}' ==> '{err['corrected_segment']}'")
        else:
            print("✓ Không phát hiện lỗi chính tả.")
        print("=" * 60)
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

            result = corrector.check_and_correct(user_input)
            print("-" * 50)
            print(f"Kết quả sửa: {result['corrected_text']}")
            if result["has_error"]:
                print(f"Chi tiết chỉnh sửa ({len(result['errors_found'])} lỗi):")
                for err in result["errors_found"]:
                    print(f"  - '{err['original_segment']}' => '{err['corrected_segment']}'")
            else:
                print("✓ Câu chuẩn, không phát hiện lỗi.")
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát chương trình.")
            break


if __name__ == "__main__":
    main()
