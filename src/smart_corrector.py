"""Smart Vietnamese Spelling Corrector Engine.
Hỗ trợ kiểm tra và sửa lỗi chính tả tiếng Việt linh hoạt:
1. Chế độ Deep Learning (ViT5 Seq2Seq) khi có PyTorch và checkpoint.
2. Chế độ Heuristic + Phonetic + Telex Decoder thông minh hoạt động ngay lập tức
   mà không đòi hỏi GPU hoặc cài đặt thư viện nặng.
"""

import difflib
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Bảng quy đổi Telex ngược (giải mã lỗi gõ dính phím)
TELEX_REVERSE = {
    # Tổ hợp nguyên âm + phím dấu giữa từ (ví dụ: sasch -> sách, hoafng -> hoàng)
    "as": "á", "af": "à", "ar": "ả", "ax": "ã", "aj": "ạ",
    "es": "é", "ef": "è", "er": "ẻ", "ex": "ẽ", "ej": "ẹ",
    "is": "í", "if": "ì", "ir": "ỉ", "ix": "ĩ", "ij": "ị",
    "os": "ó", "of": "ò", "or": "ỏ", "ox": "õ", "oj": "ọ",
    "us": "ú", "uf": "ù", "ur": "ủ", "ux": "ũ", "uj": "ụ",
    "ys": "ý", "yf": "ỳ", "yr": "ỷ", "yx": "ỹ", "yj": "ỵ",
    "aws": "ắ", "awf": "ằ", "awr": "ẳ", "awx": "ẵ", "awj": "ặ", "aw": "ă",
    "aas": "ấ", "aaf": "ầ", "aar": "ẩ", "aax": "ẫ", "aaj": "ậ", "aa": "â",
    "ees": "ế", "eef": "ề", "eer": "ể", "eex": "ễ", "eej": "ệ", "ee": "ê",
    "oos": "ố", "oof": "ồ", "oor": "ổ", "oox": "ỗ", "ooj": "ộ", "oo": "ô",
    "ows": "ớ", "owf": "ờ", "owr": "ở", "owx": "ỡ", "owj": "ợ", "ow": "ơ",
    "uws": "ứ", "uwf": "ừ", "uwr": "ử", "uwx": "ữ", "uwj": "ự", "uw": "ư",
    "dd": "đ",
}

# Bảng quy đổi VNI ngược (ví dụ: sa1ch -> sách, nghie6ng -> nghiêng)
VNI_REVERSE = {
    "a1": "á", "a2": "à", "a3": "ả", "a4": "ã", "a5": "ạ",
    "e1": "é", "e2": "è", "e3": "ẻ", "e4": "ẽ", "e5": "ẹ",
    "i1": "í", "i2": "ì", "i3": "ỉ", "i4": "ĩ", "i5": "ị",
    "o1": "ó", "o2": "ò", "o3": "ỏ", "o4": "õ", "o5": "ọ",
    "u1": "ú", "u2": "ù", "u3": "ủ", "u4": "ũ", "u5": "ụ",
    "y1": "ý", "y2": "ỳ", "y3": "ỷ", "y4": "ỹ", "y5": "ỵ",
    "a6": "â", "e6": "ê", "o6": "ô", "a8": "ă", "o7": "ơ", "u7": "ư", "d9": "đ",
}

# Các lỗi gõ đuôi telex phổ biến
TRAILING_TELEX_MAP = {
    "hocj": "học", "đocj": "đọc", "sachs": "sách", "nghiengj": "nghiêng",
    "nguowif": "người", "dduowjc": "được", "tieengs": "tiếng", "vieetj": "việt",
    "dduowngf": "đường", "nhaf": "nhà", "nguoif": "người", "hoangf": "hoàng",
}

# Từ điển sửa lỗi ngữ âm vùng miền & teencode phổ biến
COMMON_FIXES = {
    # 1. Phương ngữ ch <-> tr
    "chời": "trời", "chong": "trong", "trân thật": "chân thật", "trang": "chang",
    "chiều chuộng": "chiều chuộng", "châu báu": "châu báu",
    # 2. Phương ngữ s <-> x
    "xạch": "sạch", "xạch xẽ": "sạch sẽ", "sạch xẽ": "sạch sẽ", "xinh xắn": "xinh xắn",
    "xinh xắng": "xinh xắn", "dất": "rất", "rất": "rất", "rễ dàng": "dễ dàng",
    "giễ dàng": "dễ dàng", "gồi": "rồi", "rồi": "rồi",
    # 3. Phím lân cận QWERTY
    "xabh": "xanh", "rấr": "rất", "nón": "nón", "bón": "nón",
    # 4. Teencode & gõ sai dấu
    "hôm nai": "hôm nay", "nai": "nay",
    "hok": "học", "qúa": "quá", "quến": "quên", "thỳ": "thì", "k": "không",
    "ko": "không", "dc": "được", "đc": "được", "ng": "người", "j": "gì",
    "bit": "biết", "bik": "biết", "iu": "yêu",
    # 5. Dấu hỏi ngã
    "sữa chữa": "sửa chữa", "nghỉ ngợi": "nghĩ ngợi", "bảo bùng": "bão bùng",
    "dổ vỡ": "đổ vỡ", "nghỉ ngơi": "nghĩ ngơi",
    # 6. Cụm từ không dấu sang có dấu
    "tieng viet": "tiếng Việt", "tieng": "tiếng", "viet": "Việt",
    "hom nay": "hôm nay", "troi dep": "trời đẹp", "hoc sau": "học sâu",
    "truong dai hoc": "trường đại học", "bach khoa": "Bách Khoa",
    "ha noi": "Hà Nội", "dao tao": "đào tạo", "ki su": "kỹ sư",
    "tri tue nhan tao": "trí tuệ nhân tạo",
}


class SmartVietnameseCorrector:
    """Bộ kiểm tra và sửa lỗi chính tả tiếng Việt kết hợp."""

    def __init__(self, model_path: Optional[str] = None):
        self.ai_model = None
        self.use_ai = False

        # Thử nạp mô hình Deep Learning nếu có sẵn
        try:
            import torch
            from .models.spell_corrector import VietnameseSpellCorrector
            path = model_path or "VietAI/vit5-base"
            self.ai_model = VietnameseSpellCorrector(model_name_or_path=path)
            self.use_ai = True
        except Exception:
            self.ai_model = None
            self.use_ai = False

    def _correct_word_rule(self, word: str) -> str:
        """Sửa lỗi cho 1 từ bằng các quy luật telex, vni và ngữ âm."""
        lower = word.lower()

        # 1. Tra từ điển trực tiếp
        if lower in TRAILING_TELEX_MAP:
            fix = TRAILING_TELEX_MAP[lower]
            return fix.capitalize() if word[0].isupper() else fix

        if lower in COMMON_FIXES:
            fix = COMMON_FIXES[lower]
            return fix.capitalize() if word[0].isupper() else fix

        # 2. Xử lý lỗi dính ký tự Telex giữa từ (sasch -> sách, hoafng -> hoàng)
        w_fix = lower
        for k, v in TELEX_REVERSE.items():
            if k in w_fix:
                w_fix = w_fix.replace(k, v)

        # 3. Xử lý lỗi VNI giữa từ (sa1ch -> sách)
        for k, v in VNI_REVERSE.items():
            if k in w_fix:
                w_fix = w_fix.replace(k, v)

        # 4. Xử lý dính phím telex đuôi (nghiêngj -> nghiêng, sáchs -> sách)
        if len(w_fix) > 2 and w_fix[-1] in ["j", "s", "f", "r", "x", "w"]:
            # nếu từ đã có dấu hoàn chỉnh mà bị dính thêm phím đuôi
            candidate = w_fix[:-1]
            if any(c in "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ" for c in candidate):
                w_fix = candidate

        # 5. Phím lân cận b <-> n
        if w_fix.endswith("bh") and len(w_fix) > 2:
            w_fix = w_fix[:-2] + "nh"

        if word[0].isupper():
            w_fix = w_fix.capitalize()
        return w_fix

    def correct_rule_based(self, sentence: str) -> str:
        """Sửa lỗi toàn câu theo quy tắc, giữ nguyên dấu câu."""
        # Chuẩn hoá Unicode NFC trước
        sentence = unicodedata.normalize("NFC", sentence)

        # 1. Thay thế cụm từ trước
        lower_sent = sentence.lower()
        for phrase, fix in COMMON_FIXES.items():
            if " " in phrase and phrase in lower_sent:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                sentence = pattern.sub(fix, sentence)

        # 2. Thay thế từng từ
        tokens = sentence.split()
        corrected_tokens = []
        for token in tokens:
            m = re.match(r"^([^\w]*)(.*?)([^\w]*)$", token, re.UNICODE)
            if m:
                prefix, core, suffix = m.group(1), m.group(2), m.group(3)
            else:
                prefix, core, suffix = "", token, ""

            if core:
                fixed_core = self._correct_word_rule(core)
                corrected_tokens.append(f"{prefix}{fixed_core}{suffix}")
            else:
                corrected_tokens.append(token)

        return " ".join(corrected_tokens)

    def find_differences(self, original: str, corrected: str) -> List[Dict[str, Any]]:
        """So khớp từng từ để chỉ ra vị trí và thay đổi giữa câu gốc và câu sửa."""
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

    def check_and_correct(self, text: str) -> Dict[str, Any]:
        """Phương thức chính: Nhận câu tiếng Việt bất kỳ và trả về kết quả kiểm tra/sửa lỗi."""
        text = text.strip()
        if not text:
            return {
                "original_text": "",
                "corrected_text": "",
                "has_error": False,
                "errors_found": [],
                "engine": "none",
            }

        # Nếu có AI ViT5 thì dùng AI, nếu không thì dùng bộ Smart Rule
        if self.use_ai and self.ai_model:
            corrected = self.ai_model.correct(text)
            engine = "Deep Learning (ViT5 Transformer)"
        else:
            corrected = self.correct_rule_based(text)
            engine = "Smart Vietnamese Rule & Phonetic Engine"

        diffs = self.find_differences(text, corrected)
        return {
            "original_text": text,
            "corrected_text": corrected,
            "has_error": len(diffs) > 0,
            "errors_found": diffs,
            "engine": engine,
        }
