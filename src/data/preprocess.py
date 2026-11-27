"""Canonicalization for dataset and model inputs."""
import re
import unicodedata


def normalize_vietnamese_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text if c.isspace() or unicodedata.category(c) not in {"Cc", "Cf"})
    text = re.sub(r'["«»„“”]', '"', text)
    text = re.sub(r"['‘’ʼ]", "'", text)
    return " ".join(text.split())
