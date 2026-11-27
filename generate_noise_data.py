"""Sinh câu sạch và nhiễu từ tập train VSEC đã xác minh."""
import argparse
from pathlib import Path
import random

from src.data.noise_synthesizer import VietnameseNoiseSynthesizer
from src.data.prepare_vsec import content_keys, text_key, write_rows
from src.data.validate_split import validate
from src.postprocessing import without_accents


NOISE_GROUPS = {"full_diacritics": .4, "partial_diacritics": .3, "mixed": .3}


def augment(splits, size, probability, seed, identities=True, identity_size=None):
    if size < 0 or not 0 <= probability <= 1 or (size and probability == 0) or (identity_size is not None and identity_size < 0):
        raise ValueError("Kích thước hoặc xác suất nhiễu không hợp lệ")
    rng = random.Random(seed)
    generator = VietnameseNoiseSynthesizer(seed)
    held_out = set().union(*(content_keys(row) for name in ("val", "test") for row in splits[name]))
    held_out.update(without_accents(key) for key in tuple(held_out))
    sources = {row["target_text"]: row for row in splits["train"]}
    seen = {(text_key(row["input_text"]), text_key(row["target_text"])) for row in splits["train"]}
    augmented = []

    def add(source, noisy, kind):
        clean = source["target_text"]
        key = (text_key(noisy), text_key(clean))
        if key in seen or {without_accents(part) for part in key} & held_out:
            return False
        seen.add(key)
        augmented.append(dict(
            id=f"aug_{len(augmented)+1:06d}", parent_id=source["id"], source_text=clean,
            input_text=noisy, target_text=clean,
            noise_type=kind,
        ))
        return True

    pool = list(sources)
    if identities or identity_size:
        clean_pool = pool.copy()
        rng.shuffle(clean_pool)
        identity_count = 0
        for clean in clean_pool:
            if identity_count == identity_size:
                break
            identity_count += add(sources[clean], clean, "identity")
        if identity_size is not None and identity_count != identity_size:
            raise ValueError(f"Chỉ tạo được {identity_count}/{identity_size} câu sạch")

    remaining = size
    for index, (kind, fraction) in enumerate(NOISE_GROUPS.items()):
        count = round(size * fraction) if index < len(NOISE_GROUPS) - 1 else remaining
        remaining -= count
        generated = 0
        for _ in range(max(100, count * 30)):
            if generated == count:
                break
            clean = rng.choice(pool)
            noisy = generator.accent_variant(clean, probability, kind)
            if noisy != clean and add(sources[clean], noisy, kind):
                generated += 1
        if generated != count:
            raise ValueError(f"Chỉ sinh được {generated}/{count} mẫu {kind}")
    return augmented


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--noise_prob", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/train.jsonl"))
    parser.add_argument("--identity-size", type=int)
    parser.add_argument("--no-identities", action="store_true")
    args = parser.parse_args()
    splits, _ = validate()
    rows = augment(splits, args.size, args.noise_prob, args.seed, not args.no_identities, args.identity_size)
    write_rows(args.output, rows)
    print(f"{args.size} mẫu nhiễu và {len(rows) - args.size} câu sạch: {args.output}")


if __name__ == "__main__":
    main()
