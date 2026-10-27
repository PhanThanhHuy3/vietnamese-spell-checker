from .noise_synthesizer import VietnameseNoiseSynthesizer
from .preprocess import normalize_vietnamese_text

try:
    from .dataset import SpellingDataset, build_data_collator, load_and_cache_dataset
    __all__ = [
        "normalize_vietnamese_text",
        "VietnameseNoiseSynthesizer",
        "SpellingDataset",
        "load_and_cache_dataset",
        "build_data_collator",
    ]
except ImportError:
    __all__ = [
        "normalize_vietnamese_text",
        "VietnameseNoiseSynthesizer",
    ]

