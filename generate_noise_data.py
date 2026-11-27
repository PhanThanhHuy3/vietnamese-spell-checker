"""Sinh câu sạch và nhiễu từ tập train VSEC đã xác minh."""
import argparse
from pathlib import Path
import random

from src.data.noise_synthesizer import VietnameseNoiseSynthesizer
from src.data.prepare_vsec import content_keys, text_key, write_rows
from src.data.validate_split import validate


def augment(splits, size, probability, seed, identities=True):
    if size < 0 or not 0 <= probability <= 1 or (size and probability == 0):
        raise ValueError("Kích thước hoặc xác suất nhiễu không hợp lệ")
    rng = random.Random(seed)
    generator = VietnameseNoiseSynthesizer(seed)
    held_out = set().union(*(content_keys(row) for name in ("val", "test") for row in splits[name]))
    sources = {row["target_text"]: row for row in splits["train"]}
    seen = {(text_key(row["input_text"]), text_key(row["target_text"])) for row in splits["train"]}
    augmented = []

    def add(source, noisy):
        clean = source["target_text"]
        key = (text_key(noisy), text_key(clean))
        if key in seen or set(key) & held_out:
            return False
        seen.add(key)
        augmented.append(dict(
            id=f"aug_{len(augmented)+1:06d}", parent_id=source["id"], source_text=clean,
            input_text=noisy, target_text=clean,
            noise_type="identity" if noisy == clean else "synthetic",
        ))
        return True

    if identities:
        for clean, source in sources.items():
            add(source, clean)
    pool = list(sources)
    generated = 0
    for _ in range(max(100, size * 30)):
        if generated == size:
            break
        clean = rng.choice(pool)
        noisy = generator.add_noise(clean, probability)
        if noisy != clean and add(sources[clean], noisy):
            generated += 1
    if generated != size:
        raise ValueError(f"Chỉ sinh được {generated}/{size} mẫu nhiễu duy nhất")
    return augmented


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--noise_prob", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/train.jsonl"))
    parser.add_argument("--no-identities", action="store_true")
    args = parser.parse_args()
    splits, _ = validate()
    rows = augment(splits, args.size, args.noise_prob, args.seed, not args.no_identities)
    write_rows(args.output, rows)
    print(f"{args.size} mẫu nhiễu và {len(rows) - args.size} câu sạch: {args.output}")


if __name__ == "__main__":
    main()
