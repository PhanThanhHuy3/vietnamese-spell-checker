from pathlib import Path
from typing import List, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class VietnameseSpellCorrector:
    """Wrapper cho mô hình Seq2Seq (ViT5) phục vụ bài toán phát hiện & sửa lỗi chính tả."""

    def __init__(
        self,
        model_name_or_path: str = "VietAI/vit5-base",
        device: torch.device | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Khởi tạo Tokenizer và Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

    def correct(
        self,
        texts: Union[str, List[str]],
        max_length: int = 128,
        num_beams: int = 4,
        prefix: str = "spelling: ",
    ) -> Union[str, List[str]]:
        """Sửa lỗi chính tả cho một câu hoặc một danh sách câu."""
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        inputs = [prefix + text for text in texts]
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        corrected = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return corrected[0] if is_single else corrected

    def save_pretrained(self, save_dir: Union[str, Path]) -> None:
        """Lưu trọng số mô hình và tokenizer vào thư mục đích."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

    @classmethod
    def from_pretrained(cls, save_dir: Union[str, Path], device: torch.device | None = None) -> "VietnameseSpellCorrector":
        """Load mô hình đã huấn luyện từ thư mục checkpoint."""
        return cls(model_name_or_path=str(save_dir), device=device)
