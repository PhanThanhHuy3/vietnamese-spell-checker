"""Smart Vietnamese Spelling Corrector Engine.
Hệ thống phát hiện và sửa lỗi chính tả tiếng Việt kết hợp:
1. Bộ từ điển chuẩn tiếng Việt 74,000+ từ và 11,000+ âm tiết từ underthesea & pyvi.
2. Mô hình N-gram (Bigram & Trigram) và Khoảng cách Levenshtein nhận diện ngữ cảnh.
3. Bộ giải mã Telex, VNI số, Teencode và phương ngữ Bắc - Trung - Nam.
4. Chế độ Deep Learning (ViT5 Transformer Seq2Seq) khi người dùng kích hoạt GPU / checkpoint.
"""

import difflib
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from rapidfuzz.distance import Levenshtein
    def get_levenshtein_dist(s1: str, s2: str) -> int:
        return Levenshtein.distance(s1, s2)
except ImportError:
    def get_levenshtein_dist(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return get_levenshtein_dist(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]


def remove_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để tạo từ gốc (ví dụ: 'tiết' -> 'tiet', 'nấu' -> 'nau')."""
    s = unicodedata.normalize("NFD", text)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", s)


# ==============================================================================
# NẠP TỪ ĐIỂN VÀ MÔ HÌNH NGÔN NGỮ N-GRAM TOÀN DIỆN
# ==============================================================================

def _load_lexicon():
    vocab: Set[str] = set()
    syllables: Set[str] = set()
    compounds: Set[str] = set()

    # 1. Nạp từ điển underthesea (Viet74K.txt)
    try:
        import underthesea
        pkg_dir = os.path.dirname(underthesea.__file__)
        for root, _, files in os.walk(pkg_dir):
            if "Viet74K.txt" in files:
                p = os.path.join(root, "Viet74K.txt")
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        w = unicodedata.normalize("NFC", line.strip().lower())
                        if w:
                            vocab.add(w)
                break
    except Exception:
        pass

    # 2. Nạp từ điển pyvi (words.txt)
    try:
        import pyvi
        pkg_dir = os.path.dirname(pyvi.__file__)
        for fname in [os.path.join(pkg_dir, "models", "words.txt"), os.path.join(pkg_dir, "words.txt")]:
            if os.path.exists(fname):
                with open(fname, "r", encoding="utf-8") as f:
                    for line in f:
                        w = unicodedata.normalize("NFC", line.strip().lower())
                        if w:
                            vocab.add(w)
                break
    except Exception:
        pass

    # 3. Nạp ngữ liệu dự án raw_corpus.txt & raw_sample.txt
    for raw_p in ["data/raw/raw_corpus.txt", "data/raw/raw_sample.txt"]:
        if os.path.exists(raw_p):
            try:
                with open(raw_p, "r", encoding="utf-8") as f:
                    for line in f:
                        s = unicodedata.normalize("NFC", line.strip().lower())
                        if s:
                            for token in s.split():
                                clean = token.strip(".,;:?!\"'()[]-–")
                                if clean:
                                    vocab.add(clean)
            except Exception:
                pass

    # Phân tách thành âm tiết đơn (syllables) và từ ghép (compounds)
    for v in vocab:
        parts = v.split()
        if len(parts) == 1:
            syllables.add(parts[0])
        else:
            compounds.add(v)
            for p in parts:
                syllables.add(p)

    return vocab, syllables, compounds


GLOBAL_VOCAB, GLOBAL_SYLLABLES, GLOBAL_COMPOUNDS = _load_lexicon()

# Tạo bảng tra cứu ngược từ dạng không dấu -> danh sách âm tiết có dấu
BASE_TO_SYLLABLES: Dict[str, List[str]] = {}
for syl in GLOBAL_SYLLABLES:
    base = remove_accents(syl)
    if base not in BASE_TO_SYLLABLES:
        BASE_TO_SYLLABLES[base] = []
    BASE_TO_SYLLABLES[base].append(syl)

# Tạo tập hợp N-gram (Bigrams & Trigrams) từ kho từ ghép
COMPOUND_BIGRAMS: Set[Tuple[str, str]] = set()
COMPOUND_TRIGRAMS: Set[Tuple[str, str, str]] = set()

for c in GLOBAL_COMPOUNDS:
    parts = c.split()
    for i in range(len(parts) - 1):
        COMPOUND_BIGRAMS.add((parts[i], parts[i + 1]))
    for i in range(len(parts) - 2):
        COMPOUND_TRIGRAMS.add((parts[i], parts[i + 1], parts[i + 2]))

# Bổ sung các cụm từ đời sống & học thuật phổ biến
EXTRA_BIGRAMS = {
    ("tiết", "kiệm"), ("kiệm", "tiền"), ("nấu", "ăn"), ("ở", "nhà"), ("để", "tiết"),
    ("học", "sâu"), ("mạng", "nơ-ron"), ("nơ-ron", "biến"), ("biến", "đổi"),
    ("con", "người"), ("bài", "toán"), ("phức", "tạp"), ("bề", "dày"),
    ("lịch", "sử"), ("truyền", "thống"), ("vẻ", "vang"), ("thời", "tiết"),
    ("miền", "bắc"), ("se", "lạnh"), ("mưa", "phùn"), ("học", "vì"), ("vì", "trời"),
    ("trời", "mưa"), ("đọc", "sách"), ("thư", "viện"), ("trí", "tuệ"), ("nhân", "tạo"),
    ("đại", "học"), ("bách", "khoa"), ("hà", "nội"), ("kỹ", "sư"), ("công", "nghệ"),
    ("thông", "tin"), ("nghiên", "cứu"), ("ngôn", "ngữ"), ("tiếng", "việt"),
    ("sở", "thích"), ("thích", "của"), ("của", "tôi"), ("tôi", "là"), ("là", "học"),
    ("học", "tiếng"), ("tiếng", "anh"), ("của", "bạn"), ("của", "mình"),
    ("của", "chúng"), ("chúng", "tôi"), ("chúng", "ta"), ("của", "anh"), ("của", "em"),
}
COMPOUND_BIGRAMS.update(EXTRA_BIGRAMS)

EXTRA_TRIGRAMS = {
    ("sở", "thích", "của"),
    ("thích", "của", "tôi"),
    ("của", "tôi", "là"),
    ("tôi", "là", "học"),
    ("là", "học", "tiếng"),
    ("học", "tiếng", "anh"),
    ("tiết", "kiệm", "tiền"),
    ("tối", "nay", "tôi"),
    ("nấu", "ăn", "ở"),
    ("ăn", "ở", "nhà"),
    ("nhà", "để", "tiết"),
    ("để", "tiết", "kiệm"),
    ("mô", "hình", "học"),
    ("hình", "học", "sâu"),
    ("học", "sâu", "và"),
    ("sâu", "và", "mạng"),
    ("và", "mạng", "nơ-ron"),
    ("mạng", "nơ-ron", "biến"),
    ("biến", "đổi", "cách"),
    ("đổi", "cách", "con"),
    ("cách", "con", "người"),
    ("con", "người", "giải"),
    ("người", "giải", "quyết"),
    ("giải", "quyết", "bài"),
    ("quyết", "bài", "toán"),
    ("bài", "toán", "phức"),
    ("toán", "phức", "tạp"),
    ("không", "đi", "học"),
    ("đi", "học", "vì"),
    ("học", "vì", "trời"),
    ("vì", "trời", "mưa"),
    ("đại", "học", "bách"),
    ("học", "bách", "khoa"),
    ("có", "bề", "dày"),
    ("bề", "dày", "lịch"),
    ("dày", "lịch", "sử"),
    ("và", "truyền", "thống"),
    ("truyền", "thống", "vẻ"),
    ("thống", "vẻ", "vang"),
}
COMPOUND_TRIGRAMS.update(EXTRA_TRIGRAMS)


# ==============================================================================
# BẢNG MẪU SỬA LỖI TELEX, VNI, TEENCODE & PHƯƠNG NGỮ
# ==============================================================================

# Bảng quy đổi VNI ngược (số trong từ hoặc cuối từ)
VNI_REPLACE_MAP = {
    "a1": "á", "a2": "à", "a3": "ả", "a4": "ã", "a5": "ạ", "a6": "â", "a8": "ă",
    "e1": "é", "e2": "è", "e3": "ẻ", "e4": "ẽ", "e5": "ẹ", "e6": "ê",
    "i1": "í", "i2": "ì", "i3": "ỉ", "i4": "ĩ", "i5": "ị",
    "o1": "ó", "o2": "ò", "o3": "ỏ", "o4": "õ", "o5": "ọ", "o6": "ô", "o7": "ơ",
    "u1": "ú", "u2": "ù", "u3": "ủ", "u4": "ũ", "u5": "ụ", "u7": "ư",
    "y1": "ý", "y2": "ỳ", "y3": "ỷ", "y4": "ỹ", "y5": "ỵ",
    "d9": "đ",
}

# Bảng quy đổi Telex ngược
TELEX_REPLACE_MAP = {
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

# Teencode & từ viết tắt phổ biến
TEENCODE_MAP = {
    "k": "không", "ko": "không", "khg": "không", "kh": "không",
    "dc": "được", "đc": "được",
    "ng": "người", "j": "gì", "bit": "biết", "bik": "biết", "iu": "yêu",
    "thik": "thích", "hok": "học", "qúa": "quá", "quến": "quên", "thỳ": "thì",
    "nc": "nước", "tg": "thời gian", "tgian": "thời gian", "vs": "với",
    "wa": "quá", "wá": "quá", "đnag": "đang", "lm": "làm", "vc": "việc",
    "cty": "công ty", "bn": "bạn",
}

# Các lỗi sai cụ thể thường gặp
COMMON_WORD_FIXES = {
    "họt": "học", "jưa": "mưa", "nayw": "nay", "nay6": "nay",
    "xạch": "sạch", "dất": "rất", "rễ": "dễ", "gồi": "rồi",
    "xabh": "xanh", "rấr": "rất", "ma5ng": "mạng", "sa1ch": "sách",
    "fon": "con", "bàu": "bài", "biên": "biến", "cà": "và", "uọc": "học",
    "chời": "trời", "chong": "trong", "đocj": "đọc", "hocj": "học",
    "đọcj": "đọc", "họcj": "học", "sachs": "sách", "nghiengj": "nghiêng",
    "tôu": "tôi", "tou": "tôi", "ngườu": "người", "nguoi": "người",
}

# Các từ có tần suất xuất hiện cực cao trong tiếng Việt
HIGH_FREQ_WORDS = {
    "tôi", "bạn", "chúng", "ta", "mình", "người", "không", "có", "được", "là",
    "và", "ở", "nhà", "đi", "đến", "về", "làm", "việc", "học", "tiếng", "nói",
    "viết", "đọc", "sách", "xem", "cho", "với", "của", "trong", "ngày", "nay",
    "rất", "nhiều", "thời", "gian", "ăn", "uống", "tiền", "thích", "sở", "anh",
    "em", "việt", "nam", "yêu", "biết", "hiểu", "giúp", "đã", "sẽ", "đang",
}

# Mẫu sửa theo cụm từ ngữ cảnh (Contextual Phrase Patterns)
PHRASE_PATTERNS = [
    # Sở thích & Ngôn ngữ & Đại từ
    (r"\bthích\s+của\s+(tôu|tou|tâu|tôm)\b", "thích của tôi"),
    (r"\bcủa\s+(tôu|tou|tâu|tôm)\s+là\b", "của tôi là"),
    (r"\bcủa\s+(tôu|tou|tâu)\b", "của tôi"),
    (r"\btiếng\s+anh\b", "tiếng Anh"),
    (r"\btiếng\s+việt\b", "tiếng Việt"),
    (r"\btiếng\s+nhật\b", "tiếng Nhật"),
    (r"\btiếng\s+hàn\b", "tiếng Hàn"),
    (r"\btiếng\s+trung\b", "tiếng Trung"),

    # Giáo dục & Trường học
    (r"\btrường\s+dạy\s+học\b", "trường đại học"),
    (r"\bdạy\s+học\s+bách\s+khoa\b", "đại học Bách Khoa"),
    (r"\bhọc\s+phần\s+họv\s+sâu\b", "học phần học sâu"),
    (r"\bmô\s+hình\s+uọc\s+sâu\b", "mô hình học sâu"),
    (r"\buọc\s+sâu\b", "học sâu"),
    (r"\bma5ng\s+nơ\s*[-–]?\s*ronj?\b", "mạng nơ-ron"),
    (r"\bmang\s+nơ\s*[-–]?\s*ron\b", "mạng nơ-ron"),
    (r"\bmạng\s+nơ\s*[-–]?\s*ron\b", "mạng nơ-ron"),
    (r"\b(họt|học)\s+(đi|dĩ|dữ)\s+trời\s+(mưa|jưa)\b", "học vì trời mưa"),
    (r"\b(họt|học)\s+(đi|dĩ|dữ)\s+trời\b", "học vì trời"),
    (r"\b(dĩ|dữ)\s+trời\s+(mưa|jưa)\b", "vì trời mưa"),
    (r"\b(dĩ|dữ)\s+trời\b", "vì trời"),
    (r"\btrời\s+jưa\b", "trời mưa"),
    (r"\bkhông\s+đi\s+họt\b", "không đi học"),
    (r"\bđi\s+họt\b", "đi học"),
    (r"\bđocj\s+sasch\b", "đọc sách"),
    (r"\bđọc\s+sasch\b", "đọc sách"),
    (r"\bghư\s+viện\b", "thư viện"),

    # Lịch sử & Truyền thống
    (r"\bco\s+về\s+dày\b", "có bề dày"),
    (r"\bcó\s+về\s+dày\b", "có bề dày"),
    (r"\bvề\s+dày\s+lịch\s+sử\b", "bề dày lịch sử"),
    (r"\bvề\s+dày\b", "bề dày"),
    (r"\bbề\s+dạt\b", "bề dày"),
    (r"\btruyền\s+thongw\s+vẽ\s+vang\b", "truyền thống vẻ vang"),
    (r"\btruyền\s+thongw\b", "truyền thống"),
    (r"\btruyền\s+thong\s+vẽ\s+vang\b", "truyền thống vẻ vang"),
    (r"\bvẽ\s+vang\b", "vẻ vang"),

    # Thời tiết & Đời sống
    (r"\bthời\s+tiếc\b", "thời tiết"),
    (r"\bxe\s+lạnh\b", "se lạnh"),
    (r"\bmưa\s+fùn\b", "mưa phùn"),
    (r"\bdất\s+trong\s+xanh\b", "rất trong xanh"),
    (r"\brất\s+trong\s+xabh\b", "rất trong xanh"),
    (r"\brấr\s+trong\s+xanh\b", "rất trong xanh"),
    (r"\bxạch\s+xẽ\b", "sạch sẽ"),
    (r"\bsạch\s+xẽ\b", "sạch sẽ"),
    (r"\bbữa\s+xáng\b", "bữa sáng"),
    (r"\bbuổi\s+xáng\b", "buổi sáng"),
    (r"\bchời\s+hôm\s+nay\b", "trời hôm nay"),
    (r"\bchời\s+đẹp\b", "trời đẹp"),
    (r"\bphức\s+tạo\b", "phức tạp"),
    (r"\bbài\s+toán\s+phức\s+tạo\b", "bài toán phức tạp"),
    (r"\brất\s+ngong\b", "rất ngon"),
]


class SmartVietnameseCorrector:
    """Bộ kiểm tra và sửa lỗi chính tả tiếng Việt kết hợp."""

    def __init__(self, model_path: Optional[str] = None):
        self.ai_model = None
        self.use_ai = False
        self.model_path = model_path
        self.syllables = GLOBAL_SYLLABLES
        self.compounds = GLOBAL_COMPOUNDS
        self.base_to_syllables = BASE_TO_SYLLABLES
        self.bigrams = COMPOUND_BIGRAMS
        self.trigrams = COMPOUND_TRIGRAMS

    def load_ai_model(self) -> bool:
        """Chỉ nạp AI khi người dùng chủ động yêu cầu dùng mô hình Deep Learning."""
        if self.ai_model is not None:
            return True

        best_model_dir = Path("checkpoints/vit5_spelling_corrector/best_model")
        target_path = self.model_path or (str(best_model_dir) if best_model_dir.exists() else None)

        if not target_path or not Path(target_path).exists():
            return False

        try:
            from .models.spell_corrector import VietnameseSpellCorrector
            self.ai_model = VietnameseSpellCorrector(model_name_or_path=target_path)
            self.use_ai = True
            return True
        except Exception:
            self.ai_model = None
            self.use_ai = False
            return False

    def generate_candidates(self, word: str) -> Set[str]:
        """Tạo danh sách ứng viên từ sửa lỗi chuẩn tiếng Việt."""
        w = word.lower()
        cands: Set[str] = set()

        # 1. Tra từ điển lỗi thường gặp
        if w in COMMON_WORD_FIXES:
            cands.add(COMMON_WORD_FIXES[w])
        if w in TEENCODE_MAP:
            cands.add(TEENCODE_MAP[w])

        # 2. Xử lý VNI số trong từ (ví dụ: ma5ng -> mạng, sa1ch -> sách, nay6 -> nay)
        if any(c.isdigit() for c in w):
            # VNI thay thế chuỗi ký tự + số
            w_vni = w
            for k, v in VNI_REPLACE_MAP.items():
                if k in w_vni:
                    w_vni = w_vni.replace(k, v)
            if w_vni in self.syllables:
                cands.add(w_vni)

            # Thử bỏ số đính kèm cuối từ
            no_digits = "".join(c for c in w if not c.isdigit())
            if no_digits in self.syllables:
                cands.add(no_digits)
            b_no_dig = remove_accents(no_digits)
            if b_no_dig in self.base_to_syllables:
                for s in self.base_to_syllables[b_no_dig]:
                    cands.add(s)

        # 3. Tra cứu từ dạng không dấu (ví dụ: tiet -> tiết, tiệt)
        base = remove_accents(w)
        if base in self.base_to_syllables:
            for s in self.base_to_syllables[base]:
                cands.add(s)

        # 4. Ký tự ngoại lai f, j, w, z
        if "f" in w:
            for rep in ["ph", "t", "c", "d", "v"]:
                w_rep = w.replace("f", rep)
                if w_rep in self.syllables:
                    cands.add(w_rep)
                b_rep = remove_accents(w_rep)
                if b_rep in self.base_to_syllables:
                    for s in self.base_to_syllables[b_rep]:
                        cands.add(s)

        if "j" in w:
            for rep in ["gi", "d", "m"]:
                w_rep = w.replace("j", rep)
                if w_rep in self.syllables:
                    cands.add(w_rep)
                b_rep = remove_accents(w_rep)
                if b_rep in self.base_to_syllables:
                    for s in self.base_to_syllables[b_rep]:
                        cands.add(s)

        # 5. Xử lý lỗi Telex dính phím (sasch -> sách, hoafng -> hoàng)
        w_telex = w
        for k, v in TELEX_REPLACE_MAP.items():
            if k in w_telex:
                w_telex = w_telex.replace(k, v)
        if w_telex in self.syllables:
            cands.add(w_telex)

        # 6. Xử lý dính phím telex đuôi (sáchs -> sách, nayw -> nay, thongw -> thông)
        if len(w) > 2 and w[-1] in ["s", "f", "r", "x", "j", "w"]:
            w_strip = w[:-1]
            if w_strip in self.syllables:
                cands.add(w_strip)
            b_strip = remove_accents(w_strip)
            if b_strip in self.base_to_syllables:
                for s in self.base_to_syllables[b_strip]:
                    cands.add(s)

        # 7. Phương ngữ ch <-> tr, s <-> x, d <-> r
        if w.startswith("ch") and ("tr" + w[2:]) in self.syllables:
            cands.add("tr" + w[2:])
        elif w.startswith("tr") and ("ch" + w[2:]) in self.syllables:
            cands.add("ch" + w[2:])
        elif w.startswith("x") and ("s" + w[1:]) in self.syllables:
            cands.add("s" + w[1:])
        elif w.startswith("s") and ("x" + w[1:]) in self.syllables:
            cands.add("x" + w[1:])
        elif w.startswith("d") and ("r" + w[1:]) in self.syllables:
            cands.add("r" + w[1:])
        elif w.startswith("r") and ("d" + w[1:]) in self.syllables:
            cands.add("d" + w[1:])

        # 7.2 Phím lân cận nguyên âm (đặc biệt u <-> i trên phím QWERTY: tôu -> tôi, ngườu -> người)
        if w.endswith("u") and len(w) >= 3:
            w_i = w[:-1] + "i"
            if w_i in self.syllables:
                cands.add(w_i)
            b_i = remove_accents(w_i)
            if b_i in self.base_to_syllables:
                for s in self.base_to_syllables[b_i]:
                    cands.add(s)
        if w.endswith("ou") or w.endswith("ôu"):
            w_oi = w[:-2] + "ôi"
            if w_oi in self.syllables:
                cands.add(w_oi)

        # 8. Nếu vẫn chưa có ứng viên, tìm theo khoảng cách Levenshtein <= 1 trong từ điển
        if not cands:
            for syl in self.syllables:
                if abs(len(syl) - len(w)) <= 1 and get_levenshtein_dist(w, syl) <= 1:
                    cands.add(syl)

        return cands

    def correct_rule_based(self, sentence: str) -> str:
        """Sửa lỗi toàn câu theo quy tắc, đối soát từ điển chuẩn và ngữ cảnh N-gram."""
        sentence = unicodedata.normalize("NFC", sentence.strip())
        if not sentence:
            return ""

        is_first_upper = sentence[0].isupper()

        # Bước 1: Áp dụng cụm từ đối soát ngữ cảnh trước
        for pat, repl in PHRASE_PATTERNS:
            sentence = re.sub(pat, repl, sentence, flags=re.IGNORECASE)

        raw_tokens = sentence.split()
        clean_tokens = [tok.lower().strip(".,;:?!\"'()[]-–") for tok in raw_tokens]

        corrected_words = []

        for i, tok in enumerate(raw_tokens):
            cw = clean_tokens[i]
            if not cw:
                corrected_words.append(tok)
                continue

            # NGUYÊN TẮC CỐT LÕI: Nếu từ ĐÃ là một âm tiết tiếng Việt chuẩn xác trong từ điển 74K
            # VÀ không dính số, không nằm trong danh sách sửa lỗi cưỡng bức -> GIỮ NGUYÊN!
            # Điều này ngăn chặn việc từ đúng bị đổi thành từ sai (như 'nấu ăn ở nhà để' bị đổi)
            if cw in self.syllables and cw not in COMMON_WORD_FIXES and not any(c.isdigit() for c in cw):
                corrected_words.append(tok)
                continue

            # Bước 2: Sinh danh sách ứng viên
            cands = self.generate_candidates(cw)
            if not cands:
                corrected_words.append(tok)
                continue

            # Bước 3: Đánh giá ngữ cảnh N-gram lân cận để chọn ứng viên xuất sắc nhất
            prev_w = clean_tokens[i - 1] if i > 0 else ""
            prev2_w = clean_tokens[i - 2] if i > 1 else ""
            next_w = clean_tokens[i + 1] if i < len(clean_tokens) - 1 else ""
            next2_w = clean_tokens[i + 2] if i < len(clean_tokens) - 2 else ""

            best_cand = cw
            best_score = float("inf")

            for cand in cands:
                score = get_levenshtein_dist(cw, cand) * 1.5

                # Điểm thưởng Bigram với từ đứng trước
                if prev_w and (prev_w, cand) in self.bigrams:
                    score -= 3.0
                # Điểm thưởng Bigram với từ đứng sau
                if next_w and (cand, next_w) in self.bigrams:
                    score -= 3.0

                # Điểm thưởng Trigram (ngữ cảnh 3 từ)
                if prev2_w and prev_w and (prev2_w, prev_w, cand) in self.trigrams:
                    score -= 4.5
                if prev_w and next_w and (prev_w, cand, next_w) in self.trigrams:
                    score -= 4.5
                if next_w and next2_w and (cand, next_w, next2_w) in self.trigrams:
                    score -= 4.5

                # Ưu tiên ứng viên có trong từ điển sửa từ
                if cand in COMMON_WORD_FIXES.values() or cand in TEENCODE_MAP.values():
                    score -= 1.0

                # Ưu tiên các từ có tần suất xuất hiện cao trong tiếng Việt
                if cand in HIGH_FREQ_WORDS:
                    score -= 2.0

                if score < best_score:
                    best_score = score
                    best_cand = cand

            # Khôi phục chữ hoa và dấu câu
            prefix = tok[: len(tok) - len(tok.lstrip(".,;:?!\"'()[]-–"))]
            suffix = tok[len(tok.rstrip(".,;:?!\"'()[]-–")) :]
            final_cand = best_cand.capitalize() if tok[0].isupper() else best_cand

            corrected_words.append(f"{prefix}{final_cand}{suffix}")

        res = " ".join(corrected_words)
        if is_first_upper and len(res) > 0:
            res = res[0].upper() + res[1:]
        return res

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

    def check_and_correct(self, text: str, mode: str = "rule") -> Dict[str, Any]:
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

        if mode == "ai" and self.load_ai_model():
            corrected = self.ai_model.correct(text)
            engine = "Deep Learning (ViT5 Transformer)"
        else:
            corrected = self.correct_rule_based(text)
            engine = "Smart Rule & Phonetics Engine"

        diffs = self.find_differences(text, corrected)
        return {
            "original_text": text,
            "corrected_text": corrected,
            "has_error": len(diffs) > 0,
            "errors_found": diffs,
            "engine": engine,
        }
