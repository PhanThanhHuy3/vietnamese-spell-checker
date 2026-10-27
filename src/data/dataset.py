import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from .preprocess import normalize_vietnamese_text


class SpellingDataset(Dataset):
    """PyTorch Dataset đọc các cặp câu (input_text, target_text) từ file JSON Lines."""

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: AutoTokenizer,
        max_source_length: int = 128,
        max_target_length: int = 128,
        prefix: str = "spelling: ",
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prefix = prefix
        self.data: List[Dict[str, str]] = self._load_data()

    def _load_data(self) -> List[Dict[str, str]]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dataset: {self.file_path}")

        samples = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # Đảm bảo trường input_text và target_text tồn tại
                input_text = normalize_vietnamese_text(item.get("input_text", ""))
                target_text = normalize_vietnamese_text(item.get("target_text", ""))

                if input_text and target_text:
                    samples.append({
                        "input_text": input_text,
                        "target_text": target_text,
                    })
        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        src_text = self.prefix + sample["input_text"]
        tgt_text = sample["target_text"]

        # Mã hóa đầu vào
        model_inputs = self.tokenizer(
            src_text,
            max_length=self.max_source_length,
            padding=False,
            truncation=True,
            return_tensors=None,
        )

        # Mã hóa nhãn đích
        labels = self.tokenizer(
            tgt_text,
            max_length=self.max_target_length,
            padding=False,
            truncation=True,
            return_tensors=None,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def build_data_collator(tokenizer: AutoTokenizer, model: Any = None) -> DataCollatorForSeq2Seq:
    """Tạo DataCollator chuẩn của HuggingFace cho Seq2Seq, hỗ trợ padding linh hoạt."""
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )


def load_and_cache_dataset(
    file_path: str | Path,
    tokenizer: AutoTokenizer,
    max_source_length: int = 128,
    max_target_length: int = 128,
) -> SpellingDataset:
    """Hàm tiện ích load dataset nhanh chóng."""
    return SpellingDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
