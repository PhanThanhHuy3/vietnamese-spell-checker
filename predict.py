"""Kiểm tra chính tả bằng cùng mô hình và định dạng kết quả với website."""
import argparse
import json
from pathlib import Path

from src.data.prepare_vsec import ROOT
from src.inference import MODEL_PATHS, CorrectionService, ModelUnavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text")
    parser.add_argument("--model", choices=("auto", "baseline", *MODEL_PATHS), default="auto")
    parser.add_argument("--model_path", type=Path)
    args = parser.parse_args()
    if args.model_path and args.model not in MODEL_PATHS:
        parser.error("Chọn vit5 hoặc byt5 khi chỉ định checkpoint")
    paths = {**MODEL_PATHS, args.model: args.model_path.resolve()} if args.model_path else None
    service = CorrectionService(ROOT, paths)

    def show(text):
        print(json.dumps(service.correct(text, args.model), ensure_ascii=False, indent=2))

    try:
        if args.text is not None:
            show(args.text)
            return
        while True:
            text = input("Nhập văn bản (exit để thoát): ")
            if text.strip().lower() == "exit":
                return
            if text.strip():
                show(text)
    except (KeyboardInterrupt, EOFError):
        return
    except ModelUnavailable as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
