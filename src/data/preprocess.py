import re
import unicodedata

# Bảng quy đổi chuẩn hoá vị trí dấu thanh tiếng Việt (chuẩn mới/chuẩn cũ)
# Mặc định chuẩn hoá về kiểu hiện đại (ví dụ: hòa, thúy, khỏe)
_BANG_NGUYEN_AM = [
    ["a", "à", "á", "ả", "ã", "ạ"],
    ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ"],
    ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ"],
    ["e", "è", "é", "ẻ", "ẽ", "ẹ"],
    ["ê", "ề", "ế", "ể", "ễ", "ệ"],
    ["i", "ì", "í", "ỉ", "ĩ", "ị"],
    ["o", "ò", "ó", "ỏ", "õ", "ọ"],
    ["ô", "ồ", "ố", "ổ", "ỗ", "ộ"],
    ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ"],
    ["u", "ù", "ú", "ủ", "ũ", "ụ"],
    ["ư", "ừ", "ứ", "ử", "ữ", "ự"],
    ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ"],
]

_BANG_DAU_THANH = ["", "f", "s", "r", "x", "j"]


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa chuỗi văn bản về định dạng chuẩn NFC (Unicode dựng sẵn)."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


def clean_whitespaces(text: str) -> str:
    """Loại bỏ khoảng trắng thừa và ký tự điều khiển ẩn."""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_vietnamese_text(text: str) -> str:
    """Quy trình tiền xử lý văn bản tiếng Việt hoàn chỉnh:
    1. Chuẩn hóa Unicode NFC
    2. Chuẩn hóa khoảng trắng & ký tự thừa
    3. Chuẩn hóa dấu ngoặc kép / đơn
    """
    if not text:
        return ""

    text = normalize_unicode(text)
    # Chuẩn hoá các loại dấu ngoặc kép
    text = re.sub(r"[\"\"«»„“]", '"', text)
    text = re.sub(r"[\'\'`ʼ]", "'", text)
    # Loại bỏ khoảng trắng thừa
    text = clean_whitespaces(text)
    return text
