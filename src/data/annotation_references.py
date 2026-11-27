"""Read VSEC reference variants from the verified source export."""
from itertools import product

from rapidfuzz.distance import Levenshtein

from src.data.prepare_vsec import RAW, file_hash, read_rows
from src.data.preprocess import normalize_vietnamese_text as normalize


def load_reference_variants(manifest):
    if not RAW.is_file() or file_hash(RAW) != manifest["raw_sha256"]:
        return {}
    result = {}
    for index, row in enumerate(read_rows(RAW), 1):
        annotations = sorted(row["annotations"], key=lambda entry: entry["id"])
        if not any(not item["is_correct"] and len(item["alternative_syllables"]) > 1 for item in annotations):
            continue
        choices = [tuple(map(normalize, item["alternative_syllables"])) if not item["is_correct"]
                   else (normalize(item["current_syllable"]),) for item in annotations]
        variants = dict.fromkeys(" ".join(filter(None, parts)) for parts in product(*choices))
        result[f"vsec_{index:05d}"] = tuple(variants)
    return result


def best_reference(prediction, canonical, alternatives):
    if not alternatives or prediction == canonical:
        return canonical
    return min(dict.fromkeys((canonical, *alternatives)),
               key=lambda option: Levenshtein.distance(prediction.split(), option.split()))
