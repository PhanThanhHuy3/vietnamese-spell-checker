"""Chuẩn bị các tập VSEC cố định trước khi tăng cường dữ liệu."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
from urllib.request import urlretrieve

from src.data.preprocess import normalize_vietnamese_text as normalize

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/VSEC.jsonl"
PROCESSED = ROOT / "data/processed"
MANIFEST = ROOT / "data/splits/split_manifest.json"
SPLITS = ("train", "val", "test")
URL = "https://raw.githubusercontent.com/VSEC2021/VSEC/main/Dataset/VSEC.jsonl"


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_json(path, data):
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, rows):
    write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def text_key(text):
    text = normalize(text).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in text).split()) or text


def content_keys(row):
    return {text_key(row[field]) for field in ("input_text", "target_text", "source_text")}


def parse_record(row, position):
    source = normalize(row["text"])
    annotations = sorted(row["annotations"], key=lambda item: item["id"])
    if not source or not annotations or len({a["id"] for a in annotations}) != len(annotations):
        raise ValueError("Thiếu văn bản hoặc annotation hợp lệ")
    inputs, targets = [], []
    for ann in annotations:
        current = normalize(ann["current_syllable"])
        target = current if ann["is_correct"] else normalize(ann["alternative_syllables"][0])
        if not current or not target or not isinstance(ann["is_correct"], bool):
            raise ValueError("Annotation không hợp lệ")
        inputs.append(current)
        targets.append(target)
    return dict(id=f"vsec_{position:05d}", source_text=source,
                input_text=" ".join(inputs), target_text=" ".join(targets),
                error_count=sum(a != b for a, b in zip(inputs, targets)))


def group_rows(rows):
    parent = list(range(len(rows)))
    owner = {}

    def root(index):
        while index != parent[index]:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, row in enumerate(rows):
        for key in sorted(content_keys(row)):
            parent[root(index)] = root(owner.setdefault(key, index))
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)
    return list(groups.values())


def prepare(raw=RAW, output=PROCESSED, manifest_path=MANIFEST, seed=42):
    rows, rejected, seen = [], [], set()
    duplicates = 0
    with raw.open(encoding="utf-8-sig") as handle:
        for position, line in enumerate(handle, 1):
            try:
                row = parse_record(json.loads(line), position)
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                rejected.append(position)
                continue
            rows.append(row)
    # Group before deduplication so even a duplicate with another raw source links its group.
    groups = []
    for group in group_rows(rows):
        unique = []
        for row in group:
            key = (row["input_text"], row["target_text"])
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            unique.append(row)
        groups.append(unique)
    if len(groups) < 3:
        raise ValueError("Cần ít nhất ba nhóm VSEC độc lập")
    random.Random(seed).shuffle(groups)
    count = len(seen)
    train, val = int(count * .8), int(count * .1)
    goals = dict(zip(SPLITS, (train, val, count - train - val)))
    allocation = {name: [] for name in SPLITS}
    for index, group in enumerate(groups):
        empty = [name for name in SPLITS if not allocation[name]]
        chosen = empty[0] if len(groups) - index == len(empty) else max(
            SPLITS, key=lambda name: goals[name] - len(allocation[name])
        )
        allocation[chosen].extend(group)
    manifest = dict(version=1, seed=seed, raw_sha256=file_hash(raw), duplicates=duplicates,
                    rejected_lines=rejected, splits={})
    for name, split in allocation.items():
        path = output / f"{name}.jsonl"
        write_rows(path, split)
        manifest["splits"][name] = dict(
            sha256=file_hash(path), ids=[row["id"] for row in split], samples=len(split),
            annotated_errors=sum(row["error_count"] for row in split),
            clean_sentences=sum(row["error_count"] == 0 for row in split),
            errors_per_sentence=dict(sorted(Counter(row["error_count"] for row in split).items())),
        )
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=PROCESSED)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.raw.exists():
        if args.raw != RAW:
            parser.error(f"Không tìm thấy {args.raw}")
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.raw.with_suffix(".download")
        urlretrieve(URL, temporary)
        temporary.replace(args.raw)
    manifest = prepare(args.raw, args.output, args.manifest, args.seed)
    print({name: info["samples"] for name, info in manifest["splits"].items()})
    print(f"Trùng: {manifest['duplicates']}. Dòng không hợp lệ: {manifest['rejected_lines']}")


if __name__ == "__main__":
    main()
