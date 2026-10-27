"""Module sinh nhiễu lỗi chính tả Tiếng Việt tự động (Vietnamese Spelling Noise Generator).

Cung cấp các phương thức mô phỏng các lỗi chính tả thực tế phổ biến:
1. Lỗi gõ phím Telex / VNI (ví dụ: sách -> sasch, nghiêng -> nghiengj, sa1ch)
2. Lỗi gõ phím lân cận trên bàn phím QWERTY (ví dụ: n -> b, a -> s)
3. Lỗi vùng miền / phương ngữ (ví dụ: ch <-> tr, s <-> x, d <-> gi <-> r, hỏi <-> ngã)
4. Lỗi bỏ dấu thanh hoặc bỏ toàn bộ dấu (ví dụ: tiếng việt -> tieng viet)
"""

import argparse
import random
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# ==============================================================================
# 1. BẢNG MÃ QUY ĐỔI TELEX & VNI
# ==============================================================================

# Quy đổi nguyên âm có dấu sang chuỗi ký tự Telex tương ứng
TELEX_MAP = {
    # a, ă, â
    "á": "as", "à": "af", "ả": "ar", "ã": "ax", "ạ": "aj",
    "ắ": "aws", "ằ": "awf", "ẳ": "awr", "ẵ": "awx", "ặ": "awj", "ă": "aw",
    "ấ": "aas", "ầ": "aaf", "ẩ": "aar", "ẫ": "aax", "ậ": "aaj", "â": "aa",
    # e, ê
    "é": "es", "è": "ef", "ẻ": "er", "ẽ": "ex", "ẹ": "ej",
    "ế": "ees", "ề": "eef", "ể": "eer", "ễ": "eex", "ệ": "eej", "ê": "ee",
    # i
    "í": "is", "ì": "if", "ỉ": "ir", "ĩ": "ix", "ị": "ij",
    # o, ô, ơ
    "ó": "os", "ò": "of", "ỏ": "or", "õ": "ox", "ọ": "oj",
    "ố": "oos", "ồ": "oof", "ổ": "oor", "ỗ": "oox", "ộ": "ooj", "ô": "oo",
    "ớ": "ows", "ờ": "owf", "ở": "owr", "ỡ": "owx", "ợ": "owj", "ơ": "ow",
    # u, ư
    "ú": "us", "ù": "uf", "ủ": "ur", "ũ": "ux", "ụ": "uj",
    "ứ": "uws", "ừ": "uwf", "ử": "uwr", "ữ": "uwx", "ự": "uwj", "ư": "uw",
    # y
    "ý": "ys", "ỳ": "yf", "ỷ": "yr", "ỹ": "yx", "ỵ": "yj",
    # đ
    "đ": "dd",
}

# Quy đổi nguyên âm có dấu sang chuỗi ký tự VNI tương ứng
VNI_MAP = {
    # a, ă, â
    "á": "a1", "à": "a2", "ả": "a3", "ã": "a4", "ạ": "a5",
    "ắ": "a81", "ằ": "a82", "ẳ": "a83", "ẵ": "a84", "ặ": "a85", "ă": "a8",
    "ấ": "a61", "ầ": "a62", "ẩ": "a63", "ẫ": "a64", "ậ": "a65", "â": "a6",
    # e, ê
    "é": "e1", "è": "e2", "ẻ": "e3", "ẽ": "e4", "ẹ": "e5",
    "ế": "e61", "ề": "e62", "ể": "e63", "ễ": "e64", "ệ": "e65", "ê": "e6",
    # i
    "í": "i1", "ì": "i2", "ỉ": "i3", "ĩ": "i4", "ị": "i5",
    # o, ô, ơ
    "ó": "o1", "ò": "o2", "ỏ": "o3", "õ": "o4", "ọ": "o5",
    "ố": "o61", "ồ": "o62", "ổ": "o63", "ỗ": "o64", "ộ": "o65", "ô": "o6",
    "ớ": "o71", "ờ": "o72", "ở": "o73", "ỡ": "o74", "ợ": "o75", "ơ": "o7",
    # u, ư
    "ú": "u1", "ù": "u2", "ủ": "u3", "ũ": "u4", "ụ": "u5",
    "ứ": "u71", "ừ": "u72", "ử": "u73", "ữ": "u74", "ự": "u75", "ư": "u7",
    # y
    "ý": "y1", "ỳ": "y2", "ỷ": "y3", "ỹ": "y4", "ỵ": "y5",
    # đ
    "đ": "d9",
}

# Các phím gõ dấu telex và vni thường bị gõ thừa hoặc dính phím
TELEX_KEYS = ["s", "f", "r", "x", "j", "w", "z"]
VNI_KEYS = ["1", "2", "3", "4", "5", "6", "7", "8"]

# ==============================================================================
# 2. BÀN PHÍM QWERTY LÂN CẬN (ADJACENT KEYS)
# ==============================================================================
QWERTY_NEIGHBORS = {
    "q": ["w", "a"],
    "w": ["q", "e", "s", "a"],
    "e": ["w", "r", "d", "s"],
    "r": ["e", "t", "f", "d"],
    "t": ["r", "y", "g", "f"],
    "y": ["t", "u", "h", "g"],
    "u": ["y", "i", "j", "h"],
    "i": ["u", "o", "k", "j"],
    "o": ["i", "p", "l", "k"],
    "p": ["o", "l"],
    "a": ["q", "w", "s", "z"],
    "s": ["w", "e", "a", "d", "z", "x"],
    "d": ["e", "r", "s", "f", "x", "c"],
    "f": ["r", "t", "d", "g", "c", "v"],
    "g": ["t", "y", "f", "h", "v", "b"],
    "h": ["y", "u", "g", "j", "b", "n"],
    "j": ["u", "i", "h", "k", "n", "m"],
    "k": ["i", "o", "j", "l", "m"],
    "l": ["o", "p", "k"],
    "z": ["a", "s", "x"],
    "x": ["z", "s", "d", "c"],
    "c": ["x", "d", "f", "v"],
    "v": ["c", "f", "g", "b"],
    "b": ["v", "g", "h", "n"],
    "n": ["b", "h", "j", "m"],  # n sát cạnh b
    "m": ["n", "j", "k"],
}

# ==============================================================================
# 3. BẢNG LỖI VÙNG MIỀN / NGỮ ÂM TIẾNG VIỆT
# ==============================================================================

# Phụ âm đầu
REGIONAL_INITIAL = {
    "ch": ["tr"],
    "tr": ["ch"],
    "s": ["x"],
    "x": ["s"],
    "d": ["gi", "r"],
    "gi": ["d", "r"],
    "r": ["d", "gi"],
    "l": ["n"],
    "n": ["l"],
    "c": ["k"],
    "k": ["c"],
    "ngh": ["ng"],
    "ng": ["ngh"],
    "gh": ["g"],
    "g": ["gh"],
    "v": ["d", "gi"],
}

# Phụ âm cuối
REGIONAL_FINAL = {
    "c": ["t"],
    "t": ["c"],
    "n": ["ng"],
    "ng": ["n"],
    "ch": ["t", "c"],
    "i": ["y"],
    "y": ["i"],
}

# Lỗi dấu hỏi <-> ngã
HOI_NGA_MAP = {
    "ả": "ã", "ã": "ả",
    "ẳ": "ẵ", "ẵ": "ẳ",
    "ẩ": "ẫ", "ẫ": "ẩ",
    "ẻ": "ẽ", "ẽ": "ẻ",
    "ể": "ễ", "ễ": "ể",
    "ỉ": "ĩ", "ĩ": "ỉ",
    "ỏ": "õ", "õ": "ỏ",
    "ổ": "ỗ", "ỗ": "ổ",
    "ở": "ỡ", "ỡ": "ở",
    "ủ": "ũ", "ũ": "ủ",
    "ử": "ữ", "ữ": "ử",
    "ỷ": "ỹ", "ỹ": "ỷ",
}

# Bỏ dấu thanh nhưng giữ nguyên âm mũ/móc
REMOVE_TONE_ONLY = {
    "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
    "ằ": "ă", "ắ": "ă", "ẳ": "ă", "ẵ": "ă", "ặ": "ă",
    "ầ": "â", "ấ": "â", "ẩ": "â", "ẫ": "â", "ậ": "â",
    "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
    "ề": "ê", "ế": "ê", "ể": "ê", "ễ": "ê", "ệ": "ê",
    "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
    "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    "ồ": "ô", "ố": "ô", "ổ": "ô", "ỗ": "ô", "ộ": "ô",
    "ờ": "ơ", "ớ": "ơ", "ở": "ơ", "ỡ": "ơ", "ợ": "ơ",
    "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
    "ừ": "ư", "ứ": "ư", "ử": "ư", "ữ": "ư", "ự": "ư",
    "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
}


# ==============================================================================
# 4. CLASS CHÍNH: VietnameseNoiseGenerator
# ==============================================================================

class VietnameseNoiseGenerator:
    """Bộ tạo lỗi chính tả tiếng Việt chân thực phục vụ huấn luyện mô hình NLP."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    # --------------------------------------------------------------------------
    # QUY LUẬT 1: Lỗi gõ phím Telex / VNI
    # --------------------------------------------------------------------------
    def corrupt_telex_vni(self, word: str) -> str:
        """Tạo lỗi bộ gõ Telex hoặc VNI:
        - sách -> sasch (gõ phím telex 's' ngay sau nguyên âm 'a')
        - nghiêng -> nghiengj (dính phím telex dấu ở cuối từ)
        - sách -> sa1ch (lỗi gõ phím VNI số 1)
        """
        if len(word) == 0:
            return word

        sub_types = ["telex_mid", "telex_tail", "vni_mid", "vni_tail"]
        chosen_type = random.choice(sub_types)

        # 1. Telex chèn giữa từ (sách -> sasch, hoàng -> hoafng)
        if chosen_type == "telex_mid":
            for i, char in enumerate(word):
                lower_char = char.lower()
                if lower_char in TELEX_MAP:
                    telex_seq = TELEX_MAP[lower_char]
                    if char.isupper():
                        telex_seq = telex_seq.capitalize()
                    return word[:i] + telex_seq + word[i + 1 :]

        # 2. VNI chèn giữa từ (sách -> sa1ch, nghiêng -> nghie6ng)
        elif chosen_type == "vni_mid":
            for i, char in enumerate(word):
                lower_char = char.lower()
                if lower_char in VNI_MAP:
                    vni_seq = VNI_MAP[lower_char]
                    if char.isupper():
                        vni_seq = vni_seq.capitalize()
                    return word[:i] + vni_seq + word[i + 1 :]

        # 3. Telex dính phím ở cuối từ (nghiêng -> nghiengj, học -> hocj, sách -> sachs)
        elif chosen_type == "telex_tail":
            key = random.choice(TELEX_KEYS)
            if random.random() < 0.6:
                # Bỏ bớt dấu để tạo chuỗi gõ dính phím telex chưa ăn dấu (vd: nghiêng -> nghiengj, học -> hocj)
                base = self.corrupt_remove_diacritics(word, mode="all" if random.random() < 0.5 else "tone_only")
                return base + key
            return word + key

        # 4. VNI dính số ở cuối từ (sách -> sach1, nghiêng -> nghieng6)
        elif chosen_type == "vni_tail":
            key = random.choice(VNI_KEYS)
            if random.random() < 0.6:
                base = self.corrupt_remove_diacritics(word, mode="all")
                return base + key
            return word + key

        # Fallback: dính phím telex đuôi
        return word + random.choice(TELEX_KEYS)

    # --------------------------------------------------------------------------
    # QUY LUẬT 2: Lỗi gõ phím lân cận trên bàn phím QWERTY
    # --------------------------------------------------------------------------
    def corrupt_keyboard(self, word: str) -> str:
        """Thay thế một ký tự bằng phím lân cận trên bàn phím QWERTY (ví dụ: n -> b)."""
        chars = list(word)
        indices = [i for i, c in enumerate(chars) if c.lower() in QWERTY_NEIGHBORS]
        if not indices:
            return word

        idx = random.choice(indices)
        char = chars[idx]
        neighbor = random.choice(QWERTY_NEIGHBORS[char.lower()])
        chars[idx] = neighbor.upper() if char.isupper() else neighbor
        return "".join(chars)

    # --------------------------------------------------------------------------
    # QUY LUẬT 3: Lỗi vùng miền / phương ngữ
    # --------------------------------------------------------------------------
    def corrupt_regional(self, word: str) -> str:
        """Mô phỏng nhầm lẫn ngữ âm vùng miền:
        - Phụ âm đầu: ch <-> tr, s <-> x, d <-> gi <-> r, l <-> n, v.v.
        - Phụ âm cuối: c <-> t, n <-> ng
        - Dấu thanh: hỏi <-> ngã
        """
        lower = word.lower()
        strategies = ["initial", "final", "hoi_nga"]
        random.shuffle(strategies)

        for st in strategies:
            # 1. Phụ âm đầu (ch <-> tr, s <-> x, ...)
            if st == "initial":
                for prefix, replacements in REGIONAL_INITIAL.items():
                    if lower.startswith(prefix):
                        rep = random.choice(replacements)
                        if word[0].isupper():
                            rep = rep.capitalize()
                        return rep + word[len(prefix) :]

            # 2. Phụ âm cuối (c <-> t, n <-> ng)
            elif st == "final":
                for suffix, replacements in REGIONAL_FINAL.items():
                    if lower.endswith(suffix) and len(lower) > len(suffix):
                        rep = random.choice(replacements)
                        return word[: -len(suffix)] + rep

            # 3. Dấu hỏi <-> ngã
            elif st == "hoi_nga":
                chars = list(word)
                for i, c in enumerate(chars):
                    c_lower = c.lower()
                    if c_lower in HOI_NGA_MAP:
                        target = HOI_NGA_MAP[c_lower]
                        chars[i] = target.upper() if c.isupper() else target
                        return "".join(chars)

        return word

    # --------------------------------------------------------------------------
    # QUY LUẬT 4: Lỗi bỏ dấu (Remove Diacritics)
    # --------------------------------------------------------------------------
    def corrupt_remove_diacritics(self, word: str, mode: str = "all") -> str:
        """Gỡ bỏ dấu tiếng Việt:
        - mode='all': Bỏ toàn bộ dấu thành chuỗi ASCII (ví dụ: tiếng việt -> tieng viet)
        - mode='tone_only': Bỏ dấu thanh, giữ nguyên âm có mũ/móc (ví dụ: tiếng -> tiêng)
        """
        if not word:
            return word

        if mode == "all":
            # Chuyển 'đ' -> 'd'
            text = word.replace("đ", "d").replace("Đ", "D")
            # Chuẩn hoá NFD để tách rời dấu
            nfd = unicodedata.normalize("NFD", text)
            # Loại bỏ toàn bộ ký tự tổ hợp dấu (Mark, nonspacing)
            clean = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
            return clean

        elif mode == "tone_only":
            chars = list(word)
            for i, c in enumerate(chars):
                c_lower = c.lower()
                if c_lower in REMOVE_TONE_ONLY:
                    target = REMOVE_TONE_ONLY[c_lower]
                    chars[i] = target.upper() if c.isupper() else target
                    return "".join(chars)
            return word

        return word

    # --------------------------------------------------------------------------
    # HÀM TẠO NHIỄU CHO TỪ ĐƠN LẺ
    # --------------------------------------------------------------------------
    def corrupt_word(
        self,
        word: str,
        error_types: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """Tạo lỗi cho một từ đơn lẻ.

        Args:
            word: Từ gốc cần làm sai
            error_types: Danh sách loại lỗi được phép dùng.
                         Hỗ trợ: 'telex_vni', 'keyboard', 'regional', 'remove_diacritics'.

        Returns:
            Tuple[corrupted_word, error_type_applied]
        """
        available_types = error_types or ["telex_vni", "keyboard", "regional", "remove_diacritics"]
        if not available_types or len(word) == 0:
            return word, "none"

        selected_type = random.choice(available_types)

        if selected_type == "telex_vni":
            new_word = self.corrupt_telex_vni(word)
        elif selected_type == "keyboard":
            new_word = self.corrupt_keyboard(word)
        elif selected_type == "regional":
            new_word = self.corrupt_regional(word)
        elif selected_type == "remove_diacritics":
            new_word = self.corrupt_remove_diacritics(word, mode=random.choice(["all", "tone_only"]))
        else:
            new_word = word

        return new_word, selected_type

    # --------------------------------------------------------------------------
    # HÀM TẠO NHIỄU CHO TOÀN BỘ CÂU
    # --------------------------------------------------------------------------
    def generate_noise(
        self,
        sentence: str,
        error_rate: float = 0.3,
        error_types: Optional[List[str]] = None,
    ) -> str:
        """Tạo các lỗi chính tả ngẫu nhiên cho một câu văn bản sạch.

        Args:
            sentence: Câu văn bản tiếng Việt sạch
            error_rate: Tỷ lệ từ bị làm nhiễu (mặc định 0.3 = 30% số từ)
            error_types: Danh sách các quy luật áp dụng (None = dùng cả 4 quy luật)

        Returns:
            Câu văn bản đã bị chèn lỗi chính tả
        """
        tokens = sentence.split()
        if not tokens:
            return ""

        corrupted_tokens = []
        for token in tokens:
            # Tách dấu câu ở đầu và cuối (ví dụ: 'sách,' -> prefix='', word='sách', suffix=',')
            m = re.match(r"^([^\w]*)(.*?)([^\w]*)$", token, re.UNICODE)
            if m:
                prefix, core_word, suffix = m.group(1), m.group(2), m.group(3)
            else:
                prefix, core_word, suffix = "", token, ""

            if core_word and random.random() < error_rate:
                new_core, _ = self.corrupt_word(core_word, error_types=error_types)
                corrupted_tokens.append(f"{prefix}{new_core}{suffix}")
            else:
                corrupted_tokens.append(token)

        return " ".join(corrupted_tokens)

    def generate_with_details(
        self,
        sentence: str,
        error_rate: float = 0.3,
        error_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Tạo lỗi câu kèm thông tin chi tiết từng từ bị thay đổi để debug/kiểm thử."""
        tokens = sentence.split()
        corrupted_tokens = []
        modifications = []

        for idx, token in enumerate(tokens):
            m = re.match(r"^([^\w]*)(.*?)([^\w]*)$", token, re.UNICODE)
            if m:
                prefix, core_word, suffix = m.group(1), m.group(2), m.group(3)
            else:
                prefix, core_word, suffix = "", token, ""

            if core_word and random.random() < error_rate:
                new_core, applied_rule = self.corrupt_word(core_word, error_types=error_types)
                if new_core != core_word:
                    modifications.append({
                        "token_index": idx,
                        "original": core_word,
                        "corrupted": new_core,
                        "rule": applied_rule,
                    })
                corrupted_tokens.append(f"{prefix}{new_core}{suffix}")
            else:
                corrupted_tokens.append(token)

        return {
            "original_sentence": sentence,
            "corrupted_sentence": " ".join(corrupted_tokens),
            "num_errors": len(modifications),
            "modifications": modifications,
        }


# ==============================================================================
# HÀM TIỆN ÍCH DÙNG NHANH NGOÀI MODULE
# ==============================================================================

def add_spelling_noise(
    sentence: str,
    error_rate: float = 0.3,
    seed: Optional[int] = None,
) -> str:
    """Hàm tiện ích tạo lỗi chính tả nhanh chóng từ một câu văn bản."""
    generator = VietnameseNoiseGenerator(seed=seed)
    return generator.generate_noise(sentence, error_rate=error_rate)


# ==============================================================================
# 5. DEMO VÀ THỬ NGHIỆM TRỰC TIẾP TỪ DÒNG LỆNH (CLI)
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script tạo lỗi chính tả tiếng Việt nhân tạo.")
    parser.add_argument("--text", type=str, default=None, help="Câu văn bản sạch cần tạo lỗi")
    parser.add_argument("--rate", type=float, default=0.4, help="Tỷ lệ gây lỗi cho mỗi từ (0.0 đến 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    args = parser.parse_args()

    gen = VietnameseNoiseGenerator(seed=args.seed)

    print("=" * 70)
    print("      DEMO CÁC QUY LUẬT SINH LỖI CHÍNH TẢ TIẾNG VIỆT (src/noise_generator.py)")
    print("=" * 70)

    # 1. Minh họa từng quy luật riêng biệt theo đúng đề bài yêu cầu
    print("\n[1] LỖI GÕ PHÍM TELEX / VNI:")
    test_telex = ["sách", "nghiêng", "học", "hoàng", "được", "người"]
    for w in test_telex:
        print(f"  • {w:10} -> {gen.corrupt_telex_vni(w)}")

    print("\n[2] LỖI GÕ PHÍM LÂN CẬN QWERTY (ví dụ: n -> b):")
    test_keyboard = ["nón", "nam", "học", "bạn", "yêu", "làm"]
    for w in test_keyboard:
        print(f"  • {w:10} -> {gen.corrupt_keyboard(w)}")

    print("\n[3] LỖI VÙNG MIỀN PHỔ BIẾN (ch <-> tr, s <-> x, d/gi/r, hỏi <-> ngã):")
    test_regional = ["chân thật", "trời", "sạch sẽ", "xinh xắn", "dễ dàng", "nghỉ ngơi", "sửa chữa"]
    for w in test_regional:
        print(f"  • {w:10} -> {gen.corrupt_regional(w)}")

    print("\n[4] LỖI BỎ DẤU (ví dụ: tiếng việt -> tieng viet):")
    test_accents = ["tiếng việt", "học sâu", "trường đại học bách khoa", "nghiên cứu khoa học"]
    for w in test_accents:
        print(f"  • {w:25} -> {gen.corrupt_remove_diacritics(w, mode='all')}")

    # 2. Minh họa tạo lỗi ngẫu nhiên trên toàn bộ câu văn bản
    sample_sentence = args.text or "Hôm nay trời rất trong xanh, tôi ngồi đọc sách nghiên cứu về tiếng Việt tại thư viện."
    print("\n" + "=" * 70)
    print("KẾT QUẢ TẠO NHIỄU TRÊN CÂU HOÀN CHỈNH:")
    print("=" * 70)
    res = gen.generate_with_details(sample_sentence, error_rate=args.rate)
    print(f"Câu gốc   : {res['original_sentence']}")
    print(f"Câu nhiễu : {res['corrupted_sentence']}")
    print(f"Số lỗi tạo: {res['num_errors']}")
    print("Chi tiết các lỗi:")
    for mod in res["modifications"]:
        print(f"  - Từ gốc '{mod['original']}' -> '{mod['corrupted']}' (Quy luật: {mod['rule']})")
    print("=" * 70)
