import difflib
from pathlib import Path
from typing import Any, Dict, List, Union

from .models.spell_corrector import VietnameseSpellCorrector
from .utils.helpers import get_device


class SpellInferencePipeline:
    """Pipeline dự đoán và phát hiện vị trí lỗi chính tả cho người dùng cuối."""

    def __init__(self, model_dir_or_name: Union[str, Path] = "VietAI/vit5-base"):
        self.device = get_device()
        self.corrector = VietnameseSpellCorrector(
            model_name_or_path=str(model_dir_or_name),
            device=self.device,
        )

    def find_differences(self, original: str, corrected: str) -> List[Dict[str, Any]]:
        """So khớp từng từ để phát hiện vị trí và sự thay đổi của từ sai -> từ đúng."""
        orig_words = original.split()
        corr_words = corrected.split()

        matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
        diffs = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                diffs.append({
                    "type": tag,
                    "original_segment": " ".join(orig_words[i1:i2]),
                    "corrected_segment": " ".join(corr_words[j1:j2]),
                    "original_indices": (i1, i2),
                    "corrected_indices": (j1, j2),
                })
        return diffs

    def predict(self, text: str, max_length: int = 128, num_beams: int = 4) -> Dict[str, Any]:
        """Sửa lỗi một câu và trả về chi tiết các vị trí lỗi."""
        corrected = self.corrector.correct(text, max_length=max_length, num_beams=num_beams)
        diffs = self.find_differences(text, corrected)

        return {
            "original_text": text,
            "corrected_text": corrected,
            "has_error": len(diffs) > 0,
            "errors_found": diffs,
        }

    def predict_batch(self, texts: List[str], max_length: int = 128, num_beams: int = 4) -> List[Dict[str, Any]]:
        """Sửa lỗi cho một danh sách câu."""
        corrected_list = self.corrector.correct(texts, max_length=max_length, num_beams=num_beams)
        results = []
        for orig, corr in zip(texts, corrected_list):
            diffs = self.find_differences(orig, corr)
            results.append({
                "original_text": orig,
                "corrected_text": corr,
                "has_error": len(diffs) > 0,
                "errors_found": diffs,
            })
        return results
