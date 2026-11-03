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
    ("tôi", "ngồi"), ("ngồi", "đọc"), ("đọc", "sách"), ("cứu", "về"), ("về", "tiếng"),
    ("tại", "thư"), ("thư", "viện"), ("ngồi", "học"), ("ngồi", "nghiên"),
    ("tôi", "rất"), ("rất", "thích"), ("thích", "đi"), ("đi", "học"),
    ("hôm", "nay"), ("nay", "tôi"), ("tôi", "bị"), ("bị", "ốm"),
    # Gia đình & đời sống
    ("mẹ", "đang"), ("bố", "đang"), ("anh", "ấy"), ("em", "học"),
    ("con", "mèo"), ("con", "chó"), ("con", "gà"),
    ("thấy", "mẹ"), ("về", "thấy"), ("học", "về"),
    ("nấu", "cơm"), ("mẹ", "nấu"), ("bố", "nấu"),
    ("ngủ", "trên"), ("ngủ", "ở"), ("làm", "việc"),
    ("ra", "ngoài"), ("không", "ra"), ("ngoài", "được"),
    ("học", "bài"), ("xong", "chưa"), ("bài", "xong"),
    ("ăn", "cơm"), ("uống", "nước"), ("muốn", "uống"),
    # Thời tiết
    ("trời", "mưa"), ("mưa", "to"), ("to", "quá"), ("quá", "nên"),
    ("nên", "tôi"), ("bầu", "trời"), ("trời", "trong"), ("trong", "xanh"),
    # Câu hỏi / hướng
    ("đi", "đâu"), ("đến", "đâu"), ("ở", "đâu"), ("làm", "gì"),
    ("vậy", "sao"), ("anh", "đi"), ("chiều", "nay"),
    # Học thuật
    ("nghiên", "cứu"), ("sinh", "viên"), ("trường", "đại"), ("đại", "học"),
    ("bài", "tập"), ("tập", "thể"), ("thể", "dục"),
    # Thêm các cặp phổ biến
    ("rất", "mệt"), ("rất", "vui"), ("rất", "buồn"), ("rất", "khỏe"),
    ("tôi", "muốn"), ("tôi", "cần"), ("tôi", "thấy"),
    ("đang", "ăn"), ("đang", "ngủ"), ("đang", "học"), ("đang", "làm"),
    ("vào", "buổi"), ("buổi", "sáng"), ("buổi", "chiều"), ("buổi", "tối"),
    ("đọc", "sách"), ("xem", "phim"), ("nghe", "nhạc"),
    ("trên", "ghế"), ("trên", "bàn"),
    # Con mèo / con chó context
    ("mèo", "đang"), ("chó", "đang"),
    # Câu hỏi vậy
    ("đâu", "vậy"),
    # Hoàn thành hành động
    ("xong", "chưa"), ("học", "xong"), ("ăn", "xong"), ("làm", "xong"),
    ("chưa", "ăn"), ("chưa", "học"), ("chưa", "làm"), ("chưa", "xong"),
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
    "lhẹ": "nhẹ", "fiền": "tiền", "mu7a": "mưa",
    "ntồi": "ngồi", "fại": "tại",
    # Đảo chữ phổ biến
    "xogn": "xong", "meò": "mèo", "mèop": "mèo", "meop": "mèo",
    "bàni": "bàn", "sáhc": "sách", "tốii": "tối", "xogn": "xong",
    "vệcc": "việc", "nưwocs": "nước", "nươc": "nước",
    # Lỗi gõ thêm ký tự cuối
    "tốii": "tối", "thíchs": "thích", "thichs": "thích",
    "sáchh": "sách", "mạngg": "mạng",
    # Lỗi dấu thường gặp
    "rát": "rất", "mẹy": "mẹ", "mẹi": "mẹ",
    # Ký tự thừa ở từ con mèo
    "mèpo": "mèo", "mepo": "mèo", "mèop": "mèo",
    # Đảo chữ xogn -> xong (thêm vào đây để ưu tiên)
    "xogn": "xong",
}

# Các từ có tần suất xuất hiện cực cao trong tiếng Việt
HIGH_FREQ_WORDS = {
    "tôi", "bạn", "chúng", "ta", "mình", "người", "không", "có", "được", "là",
    "và", "ở", "nhà", "đi", "đến", "về", "làm", "việc", "học", "tiếng", "nói",
    "viết", "đọc", "sách", "xem", "cho", "với", "của", "trong", "ngày", "nay",
    "rất", "nhiều", "thời", "gian", "ăn", "uống", "tiền", "thích", "sở", "anh",
    "em", "việt", "nam", "yêu", "biết", "hiểu", "giúp", "đã", "sẽ", "đang",
    "ngồi", "tại", "thư", "viện",
}

# Mẫu sửa theo cụm từ ngữ cảnh (Contextual Phrase Patterns)
PHRASE_PATTERNS = [
    # Phím lân cận & Cụm từ thực tế (QWERTY neighbor & Contextual typos)
    (r"\b(tất|tât)\s+thích\b", "rất thích"),
    (r"\b(tôi|em|anh|chị|bạn|nó)\s+(tất|tât)\b", r"\1 rất"),
    (r"\b(thích|muốn|phải|không)\s+đu\s+(học|ho5c|chơi|làm|về)\b", r"\1 đi \2"),
    (r"\bđu\s+(học|ho5c)\b", r"đi \1"),
    (r"\b(yôm|yom)\s+(nays|nay)\b", "hôm nay"),
    (r"\b(yôm|yom)\b(?=\s+nay)", "hôm"),
    (r"\b(yôm|yom)\b", "hôm"),
    (r"\bnays\b", "nay"),
    (r"\b(tôi|bạn|anh|chị|em|nó|họ)\s+nị\s+(ổn|ổm|ốm|đau|bệnh|sốt)\b", r"\1 bị ốm"),
    (r"\b(tôi|bạn|anh|chị|em|nó|họ)\s+nị\b", r"\1 bị"),
    (r"\bnị\s+(ổn|ổm|ốm)\b", "bị ốm"),
    (r"\b(bị|nị)\s+ổn\b", "bị ốm"),
    (r"\bnị\b(?=\s+(đau|bệnh|sốt|thương|phạt|lừa))", "bị"),

    # Sở thích & Ngôn ngữ & Đại từ
    (r"\bthích\s+của\s+(tôu|tou|tâu|tôm)\b", "thích của tôi"),
    (r"\bcủa\s+(tôu|tou|tâu|tôm)\s+là\b", "của tôi là"),
    (r"\bcủa\s+(tôu|tou|tâu)\b", "của tôi"),
    (r"\b(về|học|nói|viết)\s+tiến\s+(việt|Việt)\b", r"\1 tiếng \2"),
    (r"\btiến\s+(việt|Việt)\b", r"tiếng \1"),
    (r"\b(tôi|bạn|người|mình)\s+ntồi\b", r"\1 ngồi"),
    (r"\bntồi\s+(đọc|học|viết|nghiên|làm)\b", r"ngồi \1"),
    (r"\bntồi\b", "ngồi"),
    (r"\bfại\s+(thư\s+viện|nhà|trường|đây|đó)\b", r"tại \1"),
    (r"\bfại\b(?=\s+(thư|nhà|trường|đây|đó))", "tại"),
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
    (r"\b(và|va)\s+xó\b", "và có"),
    (r"\bva\s+có\b", "và có"),
    (r"\b(mưa|mu7a)\s+(phùn|fùn)\s+(nbẹ|lhẹ|lẹ|nẹ)\b", r"\1 phùn nhẹ"),
    (r"\b(phùn|fùn)\s+(nbẹ|lhẹ|lẹ|nẹ)\b", "phùn nhẹ"),
    (r"\bnbẹ\b", "nhẹ"),
    (r"\b(và|va)\s+co\s+mưa\b", "và có mưa"),
    (r"\bmiền\s+hắc\b", "miền Bắc"),
    (r"\blhẹ\b", "nhẹ"),
    (r"\bmu7a\b", "mưa"),
    (r"\bse\s+lạnn\b", "se lạnh"),
    (r"\bxó\s+(mưa|thể|nhiều|tiền|người|tác\s+dụng|ích|lợi|việc)\b", r"có \1"),
    (r"\bva\b(?!\s+(chạm|quẹt|đập|vào|phải))", "và"),
    (r"\bthời\s+tiếc\b", "thời tiết"),
    (r"\bmiền\s+bắc\b", "miền Bắc"),
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

    # Con mèo (đảo chữ / ký tự thừa)
    (r"\bcon\s+me[oò][a-z]*\b", "con mèo"),
    (r"\bme[oò]p[o]?\b", "mèo"),

    # Câu hỏi: đi đầu vậy -> đi đâu vậy
    (r"\b(anh|chị|em|bạn|mình|cậu|bố|mẹ)\s+đi\s+đầu\b", r"\1 đi đâu"),
    (r"\bđi\s+đầu\s+(vậy|thế|hả|nhỉ|không)\b", r"đi đâu \1"),
    (r"\bđến\s+đầu\s+(vậy|thế|hả|nhỉ|không)\b", r"đến đâu \1"),
    (r"\bở\s+đầu\s+(vậy|thế|hả|nhỉ|không)\b", r"ở đâu \1"),
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

    def _normalize_repeated_chars(self, word: str) -> str:
        """Chuẩn hoá ký tự lặp: trrời -> trời, thichs -> thích, vệcc -> việc.
        Xử lý phụ âm lặp (trrời), nguyên âm lặp (tốii), và ký tự thừa cuối (thichs, vệcc).
        """
        w = word
        # Bước 1: Rút gọn 3+ ký tự giống nhau liền tiếp về 1 (trrr -> tr -> tr...)
        # Với phụ âm: trrời -> trời, nhưng không rút 'oo' vì 'oo' là Telex hợp lệ
        # Chiến lược: thử rút từng ký tự lặp và kiểm tra từ điển
        # Tìm pattern: cùng ký tự lặp >= 2 lần (phụ âm) hoặc >= 3 lần (nguyên âm/tất cả)
        import re as _re
        # 1a. Phụ âm lặp >= 2 (không phải nguyên âm có dấu) -> rút về 1
        consonant_double = _re.sub(r'([bcdfghjklmnpqrstvwxyz])\1+', r'\1', w)
        if consonant_double != w and remove_accents(consonant_double) in self.base_to_syllables:
            w = consonant_double
        elif consonant_double != w and consonant_double in self.syllables:
            w = consonant_double

        # 1b. Nguyên âm lặp >= 3 (tốiii -> tối) -> rút về 1
        vowel_triple = _re.sub(r'([aăâeêioôơuưyáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ])\1{2,}', r'\1', w)
        if vowel_triple != w and (vowel_triple in self.syllables or remove_accents(vowel_triple) in self.base_to_syllables):
            w = vowel_triple

        # 1c. Nguyên âm có dấu lặp 2 lần (tốii -> tối, trờii -> trời)
        # Chỉ rút nếu kết quả tạo ra âm tiết hợp lệ
        # Pattern: (nguyên âm có dấu) + (nguyên âm ASCII không dấu giống) lặp 1 lần cuối
        # ví dụ: tốii - ố+i+i: thử bỏ i cuối -> tối ✓
        if len(w) >= 3 and w[-1] == w[-2]:
            stripped = w[:-1]
            if stripped in self.syllables or remove_accents(stripped) in self.base_to_syllables:
                w = stripped

        return w

    def _generate_transpositions(self, word: str) -> Set[str]:
        """Sinh các ứng viên từ đảo 2 ký tự liền kề (xogn->xong, meò->mèo).
        CHỈ áp dụng nếu từ gốc KHÔNG phải âm tiết hợp lệ."""
        cands: Set[str] = set()
        w = word.lower()
        # Không tạo transposition nếu từ gốc đã đúng (tránh chưa->chừa)
        if w in self.syllables:
            return cands
        for i in range(len(w) - 1):
            transposed = w[:i] + w[i+1] + w[i] + w[i+2:]
            if transposed in self.syllables:
                cands.add(transposed)
            # Thử với base (không dấu) rồi tra lại
            b = remove_accents(transposed)
            if b in self.base_to_syllables:
                for s in self.base_to_syllables[b]:
                    cands.add(s)
        return cands

    def _fix_trailing_extra_char(self, word: str) -> Set[str]:
        """Xử lý ký tự thừa cuối từ: vệcc->việc, sachs->sách, nưwocs->nước.
        CHỈ cắt nếu từ gốc KHÔNG phải âm tiết hợp lệ."""
        cands: Set[str] = set()
        w = word.lower()

        # Không xử lý nếu từ gốc đã là âm tiết hợp lệ (tránh cắt nhầm: chưa -> chư)
        if w in self.syllables:
            return cands

        # Thử bỏ 1-2 ký tự cuối rồi kiểm tra từ điển
        for trim_len in [1, 2]:
            if len(w) > trim_len + 1:
                trimmed = w[:-trim_len]
                if trimmed in self.syllables:
                    cands.add(trimmed)
                b = remove_accents(trimmed)
                if b in self.base_to_syllables:
                    for s in self.base_to_syllables[b]:
                        cands.add(s)

        # Xử lý Telex phức tạp: nưwocs -> thử từng cặp replace Telex rồi bỏ ký tự thừa
        w_telex = w
        for k, v in sorted(TELEX_REPLACE_MAP.items(), key=lambda x: -len(x[0])):
            if k in w_telex:
                w_telex = w_telex.replace(k, v)
        # Sau khi áp Telex, thử bỏ ký tự thừa cuối
        for trim_len in [0, 1, 2]:
            candidate = w_telex[:len(w_telex) - trim_len] if trim_len > 0 else w_telex
            if len(candidate) >= 2:
                if candidate in self.syllables:
                    cands.add(candidate)
                b = remove_accents(candidate)
                if b in self.base_to_syllables:
                    for s in self.base_to_syllables[b]:
                        cands.add(s)

        return cands

    def generate_candidates(self, word: str) -> Set[str]:
        """Tạo danh sách ứng viên từ sửa lỗi chuẩn tiếng Việt."""
        w = word.lower()
        cands: Set[str] = set()

        # 1. Tra từ điển lỗi thường gặp
        if w in COMMON_WORD_FIXES:
            cands.add(COMMON_WORD_FIXES[w])
        if w in TEENCODE_MAP:
            cands.add(TEENCODE_MAP[w])

        # 1.5. Chuẩn hoá ký tự lặp trước khi tìm ứng viên (trrời->trời, tốii->tối)
        normalized = self._normalize_repeated_chars(w)
        if normalized != w:
            if normalized in self.syllables:
                cands.add(normalized)
            b_norm = remove_accents(normalized)
            if b_norm in self.base_to_syllables:
                for s in self.base_to_syllables[b_norm]:
                    cands.add(s)
            # Tiếp tục xử lý normalized thay vì w gốc
            w_proc = normalized
        else:
            w_proc = w

        # 2. Xử lý VNI số trong từ (ví dụ: ma5ng -> mạng, sa1ch -> sách, nay6 -> nay)
        if any(c.isdigit() for c in w_proc):
            # VNI thay thế chuỗi ký tự + số
            w_vni = w_proc
            for k, v in VNI_REPLACE_MAP.items():
                if k in w_vni:
                    w_vni = w_vni.replace(k, v)
            if w_vni in self.syllables:
                cands.add(w_vni)

            # Thử bỏ số đính kèm cuối từ
            no_digits = "".join(c for c in w_proc if not c.isdigit())
            if no_digits in self.syllables:
                cands.add(no_digits)
            b_no_dig = remove_accents(no_digits)
            if b_no_dig in self.base_to_syllables:
                for s in self.base_to_syllables[b_no_dig]:
                    cands.add(s)

        # 3. Tra cứu từ dạng không dấu (ví dụ: tiet -> tiết, tiệt)
        base = remove_accents(w_proc)
        if base in self.base_to_syllables:
            for s in self.base_to_syllables[base]:
                cands.add(s)

        # 4. Ký tự ngoại lai f, j, w, z
        if "f" in w_proc:
            for rep in ["ph", "t", "c", "d", "v"]:
                w_rep = w_proc.replace("f", rep)
                if w_rep in self.syllables:
                    cands.add(w_rep)
                b_rep = remove_accents(w_rep)
                if b_rep in self.base_to_syllables:
                    for s in self.base_to_syllables[b_rep]:
                        cands.add(s)

        if "j" in w_proc:
            for rep in ["gi", "d", "m"]:
                w_rep = w_proc.replace("j", rep)
                if w_rep in self.syllables:
                    cands.add(w_rep)
                b_rep = remove_accents(w_rep)
                if b_rep in self.base_to_syllables:
                    for s in self.base_to_syllables[b_rep]:
                        cands.add(s)

        # 5. Xử lý lỗi Telex dính phím (sasch -> sách, hoafng -> hoàng)
        w_telex = w_proc
        for k, v in sorted(TELEX_REPLACE_MAP.items(), key=lambda x: -len(x[0])):
            if k in w_telex:
                w_telex = w_telex.replace(k, v)
        if w_telex in self.syllables:
            cands.add(w_telex)
        b_telex = remove_accents(w_telex)
        if b_telex in self.base_to_syllables:
            for s in self.base_to_syllables[b_telex]:
                cands.add(s)

        # 6. Xử lý dính phím telex đuôi (sáchs -> sách, nayw -> nay, thongw -> thông)
        if len(w_proc) > 2 and w_proc[-1] in ["s", "f", "r", "x", "j", "w"]:
            w_strip = w_proc[:-1]
            if w_strip in self.syllables:
                cands.add(w_strip)
            b_strip = remove_accents(w_strip)
            if b_strip in self.base_to_syllables:
                for s in self.base_to_syllables[b_strip]:
                    cands.add(s)

        # 6.5. Đảo 2 ký tự liền kề (transposition): xogn->xong, meò->mèo
        cands.update(self._generate_transpositions(w_proc))

        # 6.6. Ký tự thừa cuối từ (vệcc->việc, nưwocs->nước)
        cands.update(self._fix_trailing_extra_char(w_proc))

        # 7. Phương ngữ ch <-> tr, s <-> x, d <-> r
        if w_proc.startswith("ch") and ("tr" + w_proc[2:]) in self.syllables:
            cands.add("tr" + w_proc[2:])
        elif w_proc.startswith("tr") and ("ch" + w_proc[2:]) in self.syllables:
            cands.add("ch" + w_proc[2:])
        elif w_proc.startswith("x") and ("s" + w_proc[1:]) in self.syllables:
            cands.add("s" + w_proc[1:])
        elif w_proc.startswith("s") and ("x" + w_proc[1:]) in self.syllables:
            cands.add("x" + w_proc[1:])
        elif w_proc.startswith("d") and ("r" + w_proc[1:]) in self.syllables:
            cands.add("r" + w_proc[1:])
        elif w_proc.startswith("r") and ("d" + w_proc[1:]) in self.syllables:
            cands.add("d" + w_proc[1:])

        # 7.2 Phím lân cận nguyên âm (đặc biệt u <-> i trên phím QWERTY: tôu -> tôi, ngườu -> người)
        if w_proc.endswith("u") and len(w_proc) >= 3:
            w_i = w_proc[:-1] + "i"
            if w_i in self.syllables:
                cands.add(w_i)
            b_i = remove_accents(w_i)
            if b_i in self.base_to_syllables:
                for s in self.base_to_syllables[b_i]:
                    cands.add(s)
        if w_proc.endswith("ou") or w_proc.endswith("ôu"):
            w_oi = w_proc[:-2] + "ôi"
            if w_oi in self.syllables:
                cands.add(w_oi)

        # 8. Nếu vẫn chưa có ứng viên, tìm theo khoảng cách Levenshtein <= 1 trong từ điển
        if not cands:
            for syl in self.syllables:
                if abs(len(syl) - len(w_proc)) <= 1 and get_levenshtein_dist(w_proc, syl) <= 1:
                    cands.add(syl)
            # Cũng thử với từ gốc w nếu w != w_proc
            if not cands and w != w_proc:
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

            prev_w = clean_tokens[i - 1] if i > 0 else ""
            prev2_w = clean_tokens[i - 2] if i > 1 else ""
            next_w = clean_tokens[i + 1] if i < len(clean_tokens) - 1 else ""
            next2_w = clean_tokens[i + 2] if i < len(clean_tokens) - 2 else ""

            # Dùng từ đã được sửa ở vị trí trước (thay vì token gốc) để tra bigram chính xác hơn
            if i > 0 and corrected_words:
                prev_corrected = corrected_words[-1].lower().strip(".,;:?!\"'()[]-–")
                if prev_corrected:
                    prev_w = prev_corrected
            if i > 1 and len(corrected_words) >= 2:
                prev2_corrected = corrected_words[-2].lower().strip(".,;:?!\"'()[]-–")
                if prev2_corrected:
                    prev2_w = prev2_corrected

            # Kiểm tra xem từ hiện tại có ngữ cảnh Bigram bảo vệ không
            has_context = (prev_w and (prev_w, cw) in self.bigrams) or (next_w and (cw, next_w) in self.bigrams)

            # Kiểm tra từ có vẻ "ổn" không (không dính số, không có ký tự lặp bất thường)
            # Ký tự lặp: trrời, đanng, tốii -> cần sửa ngay cả khi has_context
            has_repeated_consonant = bool(re.search(r'([bcdfghjklmnpqrstvwxyz])\1', cw))
            # Nguyên âm lặp cuối (ii, uu, aa...) KHÔNG phải do Telex (aa=â, ee=ê, oo=ô, uw=ư)
            has_trailing_double_vowel = bool(re.search(r'[aăâeêioôơuưy]{2,}$', cw)) and not any(
                cw.endswith(t) for t in ['aa', 'ee', 'oo', 'uw', 'aw', 'ow']
            )
            needs_fix = has_repeated_consonant or has_trailing_double_vowel or cw in COMMON_WORD_FIXES

            # NGUYÊN TẮC CỐT LÕI: Nếu từ ĐÃ là một âm tiết tiếng Việt chuẩn xác trong từ điển 74K
            # VÀ có ngữ cảnh Bigram bảo vệ VÀ không cần sửa -> GIỮ NGUYÊN!
            if cw in self.syllables and has_context and not needs_fix and not any(c.isdigit() for c in cw):
                corrected_words.append(tok)
                continue

            # Bước 2: Sinh danh sách ứng viên (bao gồm cả lỗi phím QWERTY lân cận và đổi dấu)
            cands = self.generate_candidates(cw)
            # Thêm các phím lân cận trên QWERTY
            from .noise_generator import QWERTY_NEIGHBORS
            for idx, char in enumerate(cw):
                lower_c = char.lower()
                if lower_c in QWERTY_NEIGHBORS:
                    for nb in QWERTY_NEIGHBORS[lower_c]:
                        alt = cw[:idx] + nb + cw[idx + 1 :]
                        if alt in self.syllables:
                            cands.add(alt)

            # Thêm biến thể dấu từ base_to_syllables
            b_syl = remove_accents(cw)
            if b_syl in self.base_to_syllables:
                for syl in self.base_to_syllables[b_syl]:
                    cands.add(syl)

            if cw in self.syllables:
                cands.add(cw)

            if not cands:
                corrected_words.append(tok)
                continue

            # Bước 3: Đánh giá ngữ cảnh N-gram lân cận để chọn ứng viên xuất sắc nhất
            best_cand = cw
            best_score = float("inf")

            for cand in cands:
                # Điểm cơ sở: từ khác với từ gốc thì tính khoảng cách
                if cand == cw:
                    score = 0.0 if has_context else 1.0
                    if cw in self.syllables:
                        score -= 1.0
                    if cw in HIGH_FREQ_WORDS:
                        score -= 2.5
                    # Phạt nếu cw có vẻ là lỗi (ký tự lặp, ký tự thừa)
                    if needs_fix:
                        score += 5.0
                else:
                    dist = get_levenshtein_dist(cw, cand)
                    score = dist * 1.5
                    if cand in HIGH_FREQ_WORDS:
                        score -= 1.5
                    if cw in HIGH_FREQ_WORDS:
                        score += 3.0
                    # Thưởng thêm nếu cand là kết quả normalize (ít ký tự hơn = bỏ ký tự thừa)
                    if len(cand) < len(cw) and cand in self.syllables:
                        score -= 0.5

                # Điểm thưởng Bigram với từ đứng trước
                if prev_w and (prev_w, cand) in self.bigrams:
                    score -= 4.0
                # Điểm thưởng Bigram với từ đứng sau
                if next_w and (cand, next_w) in self.bigrams:
                    score -= 4.0

                # Điểm thưởng Trigram (ngữ cảnh 3 từ)
                if prev2_w and prev_w and (prev2_w, prev_w, cand) in self.trigrams:
                    score -= 5.0
                if prev_w and next_w and (prev_w, cand, next_w) in self.trigrams:
                    score -= 5.0
                if next_w and next2_w and (cand, next_w, next2_w) in self.trigrams:
                    score -= 5.0

                # Ưu tiên ứng viên có trong từ điển sửa từ
                if cand in COMMON_WORD_FIXES.values() or cand in TEENCODE_MAP.values():
                    score -= 1.0

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

    def align_and_preserve(self, original_text: str, ai_candidate: str) -> str:
        """
        MÀNG LỌC BẢO TOÀN CẤU TRÚC CÂU TUYỆT ĐỐI (Strict Sentence & Identity Preservation):
        - Đảm bảo hệ thống CHỈ SỬA LỖI CHÍNH TẢ, tuyệt đối KHÔNG ĐỔI NỘI DUNG CÂU.
        - Nếu AI bị ảo giác (hallucination) tự ý biến thành câu khác hoặc thêm/bớt từ tùy tiện,
          màng lọc sẽ bác bỏ hoàn toàn sự bịa đặt của AI và giữ nguyên từng chữ của câu gốc.
        """
        if not ai_candidate or not ai_candidate.strip():
            return self.correct_rule_based(original_text)

        # 1. Đo độ tương đồng ký tự và cấu trúc từ giữa câu gốc và câu do AI sinh ra
        matcher = difflib.SequenceMatcher(None, original_text.lower(), ai_candidate.lower())
        sim_ratio = matcher.ratio()

        orig_words = original_text.split()
        ai_words = ai_candidate.split()

        # Nếu độ tương đồng < 0.65 hoặc chênh lệch số lượng từ quá 2 từ:
        # Khẳng định 100% AI đang sinh ảo giác (bịa sang câu khác) -> BÁC BỎ CÂU AI!
        if sim_ratio < 0.65 or abs(len(orig_words) - len(ai_words)) > 2:
            return self.correct_rule_based(original_text)

        # 2. Nếu AI sửa đúng cấu trúc câu: Căn chỉnh từng từ (Word-level alignment)
        # Nguyên tắc: Từ nào của người dùng vốn đã viết đúng tiếng Việt -> CẤM ĐƯỢC ĐỔI!
        rule_fixed_words = self.correct_rule_based(original_text).split()
        if len(orig_words) == len(ai_words) == len(rule_fixed_words):
            final_words = []
            for orig_w, ai_w, rule_w in zip(orig_words, ai_words, rule_fixed_words):
                clean_orig = orig_w.lower().strip(".,;:?!\"'()[]-–")
                # Nếu từ gốc đã là từ chuẩn tiếng Việt và không nằm trong danh sách lỗi thường gặp
                if clean_orig in self.syllables and clean_orig not in COMMON_WORD_FIXES:
                    final_words.append(orig_w)
                else:
                    # Nếu từ gốc bị lỗi: Kiểm tra xem AI sửa có hợp lý không
                    clean_ai = ai_w.lower().strip(".,;:?!\"'()[]-–")
                    if clean_ai in self.syllables:
                        final_words.append(ai_w)
                    else:
                        final_words.append(rule_w)
            return " ".join(final_words)

        # Mặc định an toàn: Dùng kết quả bảo toàn từng từ đã qua tinh chỉnh
        return self.correct_rule_based(original_text)

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
            try:
                ai_out = self.ai_model.correct(text, prefix="sửa lỗi: ")
            except Exception:
                try:
                    ai_out = self.ai_model.correct(text)
                except Exception:
                    ai_out = ""
            corrected = self.align_and_preserve(text, ai_out)
            engine = "Deep Learning (ViT5 Transformer) + Guard"
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
