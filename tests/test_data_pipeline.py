import json
import unicodedata
from pathlib import Path

from src.data.noise_synthesizer import VietnameseNoiseSynthesizer
from src.data.preprocess import normalize_vietnamese_text


def test_normalize_unicode_nfc():
    decomposed = unicodedata.normalize("NFD", "Tiếng Việt")
    assert normalize_vietnamese_text(decomposed) == unicodedata.normalize("NFC", "Tiếng Việt")


def test_normalize_whitespace_and_quotes():
    assert normalize_vietnamese_text('  “Xin   chào”  ') == '"Xin chào"'


def test_noise_synthesizer_keeps_length_for_identity_probability_zero():
    gen = VietnameseNoiseSynthesizer(seed=42)
    text = "Hôm nay trời rất đẹp."
    assert gen.add_noise(text, noise_prob=0.0) == text


def test_noise_synthesizer_is_deterministic_with_seed():
    text = "Trường đại học Bách Khoa Hà Nội có bề dày lịch sử."
    a = VietnameseNoiseSynthesizer(seed=7).add_noise(text, noise_prob=0.5)
    b = VietnameseNoiseSynthesizer(seed=7).add_noise(text, noise_prob=0.5)
    assert a == b


def test_processed_splits_are_disjoint_if_present():
    root = Path(__file__).resolve().parents[1]
    paths = {name: root / "data" / "processed" / f"{name}.jsonl" for name in ("train", "val", "test")}
    if not all(path.exists() for path in paths.values()):
        return
    ids = {}
    for name, path in paths.items():
        ids[name] = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])
