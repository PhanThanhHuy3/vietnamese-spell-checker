"""Vietnamese spelling corruption shared by the trainer and web demo."""
import random
import unicodedata

from src.postprocessing import WORD, apply_edits, is_protected, protected_spans, without_accents

TONE_MARKS = "\u0300\u0301\u0303\u0309\u0323"
TELEX_MARKS = dict(zip(TONE_MARKS + "\u0306\u031b", "fsxrjww"))
INITIALS = {"tr": "ch", "ch": "tr", "ngh": "ng", "ng": "ngh", "gi": "d", "d": "r",
            "r": "gi", "s": "x", "x": "s", "l": "n", "n": "l"}
FINALS = {"ng": "n", "n": "ng", "nh": "n", "t": "c", "c": "t"}
ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


NOISE_LABELS = {
    "mixed": "Lỗi hỗn hợp", "full_diacritics": "Không dấu toàn câu",
    "partial_diacritics": "Thiếu dấu một phần", "tone": "Sai dấu thanh",
    "initial": "Phụ âm đầu", "final": "Phụ âm cuối", "vowel": "Sai nguyên âm",
    "keyboard": "Gõ nhầm phím", "delete": "Thiếu ký tự", "insert": "Thừa ký tự",
    "transpose": "Đảo ký tự", "telex": "Telex", "vni": "VNI",
}


def to_vni(word):
    output, tones = [], []
    modifiers = dict(zip("\u0302\u031b\u0306", "678"))
    digits = dict(zip(TONE_MARKS, "12534"))
    for char in unicodedata.normalize("NFD", word):
        if char in digits:
            tones.append(digits[char])
        else:
            output.append(modifiers.get(char, {"đ": "d9", "Đ": "D9"}.get(char, char)))
    return "".join(output + tones)


def to_telex(word):
    decomposed = unicodedata.normalize("NFD", word)
    keys, tones = [], []
    previous = ""
    for char in decomposed:
        if char in TONE_MARKS:
            tones.append(TELEX_MARKS[char])
        elif char == "\u0302":
            keys.append(previous.lower())
        elif char in TELEX_MARKS:
            keys.append(TELEX_MARKS[char])
        else:
            keys.append({"đ": "dd", "Đ": "Dd"}.get(char, char))
            previous = char
    return "".join(keys + tones)


class VietnameseNoiseSynthesizer:
    def __init__(self, seed=None):
        self.random = random.Random(seed)

    def options(self, word, extended=False):
        decomposed = unicodedata.normalize("NFD", word)
        marks = [c for c in decomposed if c in TONE_MARKS]
        options = [("Thiếu dấu", without_accents(word))]
        if marks:
            wrong = "\u0303" if marks[0] == "\u0309" else "\u0309"
            options.append(("Sai dấu thanh", unicodedata.normalize("NFC", decomposed.replace(marks[0], wrong, 1))))
        for label, mapping, is_prefix in (("Phụ âm đầu", INITIALS, True), ("Phụ âm cuối", FINALS, False)):
            lower = word.lower()
            for key in sorted(mapping, key=len, reverse=True):
                if len(word) > len(key) and (lower.startswith(key) if is_prefix else lower.endswith(key)):
                    value = mapping[key]
                    changed = value + lower[len(key):] if is_prefix else lower[:-len(key)] + value
                    options.append((label, changed.capitalize() if word[0].isupper() else changed))
                    break
        options.append(("Telex", to_telex(word)))
        if extended:
            options.append(("VNI", to_vni(word)))
            vowels = [i for i, char in enumerate(decomposed) if char.casefold() in "aeiouy"]
            if vowels:
                at = self.random.choice(vowels)
                vowel = decomposed[at]
                new_vowel = self.random.choice([c for c in "aeiouy" if c != vowel.casefold()])
                if vowel.isupper():
                    new_vowel = new_vowel.upper()
                options.append(("Sai nguyên âm", unicodedata.normalize("NFC", decomposed[:at] + new_vowel + decomposed[at + 1:])))
        index = self.random.randrange(len(word))
        c = word[index]
        for row in ROWS:
            if c.lower() in row:
                pos = row.index(c.lower())
                adjacent = row[max(0, pos - 1):pos] + row[pos + 1:pos + 2]
                key = self.random.choice(adjacent)
                options.append(("Gõ nhầm phím", word[:index] + (key.upper() if c.isupper() else key) + word[index + 1:]))
                break
        options.append(("Thừa ký tự", word[:index] + c + word[index:]))
        options.append(("Thiếu ký tự", word[:index] + word[index + 1:]))
        swap = min(index, len(word) - 2)
        options.append(("Đảo ký tự", word[:swap] + word[swap + 1] + word[swap] + word[swap + 2:]))
        return [(kind, result) for kind, result in options if result and result != word]

    def generate(self, text, noise_prob=.25, mode=None):
        if not 0 <= noise_prob <= 1:
            raise ValueError("Xác suất phải nằm trong khoảng 0 đến 1")
        if mode is not None and mode not in NOISE_LABELS:
            raise ValueError("Loại lỗi không hợp lệ.")
        if noise_prob == 0:
            return text, []
        if mode in ("full_diacritics", "partial_diacritics", "mixed"):
            return self.accent_variant(text, noise_prob, mode, with_edits=True)
        protected = protected_spans(text)
        edits = []
        for match in WORD.finditer(text):
            word = unicodedata.normalize("NFC", match.group())
            if len(word) < 2 or not word.isalpha() or is_protected(match, protected):
                continue
            if self.random.random() < noise_prob:
                options = self.options(word, extended=mode in ("vowel", "vni"))
                if mode:
                    options = [item for item in options if item[0] == NOISE_LABELS[mode]]
                if not options:
                    continue
                label, changed = self.random.choice(options)
                edits.append(dict(start=match.start(), end=match.end(), original_segment=match.group(),
                                  corrected_segment=changed, category=label))
        return apply_edits(text, edits), edits

    def accent_variant(self, text, probability, mode, with_edits=False):
        """Build a full, partial or mixed accent variant for supervised training."""
        if not 0 <= probability <= 1:
            raise ValueError("Xác suất phải nằm trong khoảng 0 đến 1")
        if probability == 0:
            return (text, []) if with_edits else text
        protected = protected_spans(text)
        words = [match for match in WORD.finditer(text)
                 if match.group().isalpha() and not is_protected(match, protected)]
        accented = [match for match in words if without_accents(match.group()) != match.group()]
        if not accented:
            return (text, []) if with_edits else text
        if mode == "full_diacritics":
            changed = accented
        elif mode in ("partial_diacritics", "mixed"):
            if len(accented) < 2:
                return (text, []) if with_edits else text
            changed = [match for match in accented if self.random.random() < probability]
            if not changed:
                changed = [self.random.choice(accented)]
            if len(changed) == len(accented):
                changed.pop(self.random.randrange(len(changed)))
        else:
            raise ValueError(f"Nhóm tạo nhiễu không hợp lệ: {mode}")
        edits = [dict(start=match.start(), end=match.end(), original_segment=match.group(),
                      category=NOISE_LABELS["full_diacritics" if mode == "full_diacritics" else "partial_diacritics"],
                      corrected_segment=without_accents(match.group())) for match in changed]
        if mode == "mixed":
            available = [match for match in words if match not in changed and len(match.group()) >= 2]
            self.random.shuffle(available)
            for match in available:
                options = [item for item in self.options(match.group()) if item[0] != "Thiếu dấu"]
                if options:
                    label, replacement = self.random.choice(options)
                    edits.append(dict(start=match.start(), end=match.end(), original_segment=match.group(),
                                      category=label, corrected_segment=replacement))
                    break
            else:
                return (text, []) if with_edits else text
        edits.sort(key=lambda edit: edit["start"])
        result = apply_edits(text, edits)
        return (result, edits) if with_edits else result

    def add_noise(self, text, noise_prob=.25):
        return self.generate(text, noise_prob)[0]
