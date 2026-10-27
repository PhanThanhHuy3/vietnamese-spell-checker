import random
from typing import List

from .preprocess import normalize_vietnamese_text


class VietnameseNoiseSynthesizer:
    """Bộ sinh nhiễu lỗi chính tả Tiếng Việt tự động phục vụ bài toán huấn luyện mô hình.
    Mô phỏng các loại lỗi thực tế:
    1. Lỗi gõ Telex / Unikey
    2. Lỗi ngữ âm / Phương ngữ (s/x, tr/ch, d/gi/r, l/n, c/k, hỏi/ngã, v.v.)
    3. Lỗi khoảng cách bàn phím QWERTY (gõ nhầm phím lân cận)
    4. Lỗi mất dấu thanh / rụng dấu
    5. Lỗi cơ học (đảo vị trí ký tự, rụng ký tự, lặp ký tự)
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

        # 1. Bảng thay thế ngữ âm / phương ngữ
        self.phonetic_replacements = {
            "tr": ["ch"],
            "ch": ["tr"],
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
        }

        # 2. Bảng lỗi dấu hỏi <-> ngã
        self.hoi_nga_pairs = {
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

        # 3. Bảng gỡ bỏ dấu thanh (mất dấu)
        self.remove_tones = {
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

        # 4. Bàn phím QWERTY lân cận
        self.qwerty_neighbors = {
            "q": ["w", "a"], "w": ["q", "e", "s"], "e": ["w", "r", "d"],
            "r": ["e", "t", "f"], "t": ["r", "y", "g"], "y": ["t", "u", "h"],
            "u": ["y", "i", "j"], "i": ["u", "o", "k"], "o": ["i", "p", "l"],
            "p": ["o", "l"], "a": ["q", "s", "z"], "s": ["a", "d", "w", "x"],
            "d": ["s", "f", "e", "c"], "f": ["d", "g", "r", "v"],
            "g": ["f", "h", "t", "b"], "h": ["g", "j", "y", "n"],
            "j": ["h", "k", "u", "m"], "k": ["j", "l", "i"], "l": ["k", "o", "p"],
            "z": ["a", "x"], "x": ["z", "c", "s"], "c": ["x", "v", "d"],
            "v": ["c", "b", "f"], "b": ["v", "n", "g"], "n": ["b", "m", "h"],
            "m": ["n", "j"],
        }

        # 5. Lỗi gõ thừa phím telex
        self.telex_keys = ["s", "f", "r", "x", "j", "w", "a", "e", "o", "d"]

    def _corrupt_phonetics(self, word: str) -> str:
        """Thay thế các cặp âm phương ngữ dễ nhầm lẫn."""
        lower_word = word.lower()
        for k, v in self.phonetic_replacements.items():
            if lower_word.startswith(k):
                replacement = random.choice(v)
                if word[0].isupper():
                    replacement = replacement.capitalize()
                return replacement + word[len(k):]
        return word

    def _corrupt_hoi_nga(self, word: str) -> str:
        """Đổi dấu hỏi thành ngã hoặc ngược lại."""
        chars = list(word)
        for i, c in enumerate(chars):
            if c.lower() in self.hoi_nga_pairs:
                is_upper = c.isupper()
                new_c = self.hoi_nga_pairs[c.lower()]
                chars[i] = new_c.upper() if is_upper else new_c
                return "".join(chars)
        return word

    def _corrupt_remove_tone(self, word: str) -> str:
        """Mất dấu thanh."""
        chars = list(word)
        for i, c in enumerate(chars):
            if c.lower() in self.remove_tones:
                is_upper = c.isupper()
                new_c = self.remove_tones[c.lower()]
                chars[i] = new_c.upper() if is_upper else new_c
                return "".join(chars)
        return word

    def _corrupt_keyboard(self, word: str) -> str:
        """Gõ nhầm phím lân cận trên bàn phím QWERTY."""
        if len(word) < 2:
            return word
        chars = list(word)
        idx = random.randint(0, len(chars) - 1)
        c = chars[idx].lower()
        if c in self.qwerty_neighbors:
            neighbor = random.choice(self.qwerty_neighbors[c])
            chars[idx] = neighbor.upper() if chars[idx].isupper() else neighbor
        return "".join(chars)

    def _corrupt_telex_tail(self, word: str) -> str:
        """Dính đuôi phím telex do bộ gõ tiếng Việt bị lỗi (vd: 'người' -> 'nguowif')."""
        tail = random.choice(self.telex_keys)
        return word + tail

    def _corrupt_swap_or_drop(self, word: str) -> str:
        """Đảo vị trí hai ký tự hoặc làm rơi một ký tự."""
        if len(word) < 3:
            return word
        chars = list(word)
        op = random.choice(["swap", "drop", "duplicate"])
        idx = random.randint(0, len(chars) - 2)

        if op == "swap":
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif op == "drop":
            chars.pop(idx)
        elif op == "duplicate":
            chars.insert(idx, chars[idx])

        return "".join(chars)

    def corrupt_word(self, word: str) -> str:
        """Tạo lỗi cho một từ đơn lẻ theo xác suất các loại lỗi."""
        if len(word) <= 1 or not word.isalnum():
            return word

        corruption_types = [
            self._corrupt_phonetics,
            self._corrupt_hoi_nga,
            self._corrupt_remove_tone,
            self._corrupt_keyboard,
            self._corrupt_telex_tail,
            self._corrupt_swap_or_drop,
        ]

        # Chọn ngẫu nhiên 1 chiến lược làm nhiễu
        strategy = random.choice(corruption_types)
        return strategy(word)

    def add_noise(self, text: str, noise_prob: float = 0.25) -> str:
        """Thêm nhiễu ngẫu nhiên vào toàn bộ câu văn bản.

        Args:
            text: Câu văn bản tiếng Việt sạch (clean text).
            noise_prob: Xác suất một từ bị làm nhiễu (mặc định 25%).

        Returns:
            Câu văn bản bị làm nhiễu (noisy/corrupted text).
        """
        text = normalize_vietnamese_text(text)
        words = text.split()
        if not words:
            return ""

        corrupted_words = []
        for word in words:
            if random.random() < noise_prob:
                corrupted_words.append(self.corrupt_word(word))
            else:
                corrupted_words.append(word)

        return " ".join(corrupted_words)
