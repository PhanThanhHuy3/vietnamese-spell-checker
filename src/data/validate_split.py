"""Kiểm tra danh sách ID, chữ ký tệp và nội dung giữa các tập VSEC."""
import argparse
import json
from pathlib import Path

from src.data.near_duplicates import near_pairs
from src.data.prepare_vsec import MANIFEST, PROCESSED, SPLITS, content_keys, file_hash, read_rows, text_key


def validate(directory=PROCESSED, manifest_path=MANIFEST):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["version"] != 2:
        raise ValueError("Phiên bản manifest VSEC không được hỗ trợ")
    all_ids, owners, splits = set(), {}, {}
    for name in SPLITS:
        path = directory / f"{name}.jsonl"
        rows = read_rows(path)
        expected = manifest["splits"][name]
        if not rows or file_hash(path) != expected["sha256"] or [row["id"] for row in rows] != expected["ids"]:
            raise ValueError(f"Tập {name} không khớp manifest")
        for row in rows:
            if row["id"] in all_ids:
                raise ValueError(f"ID trùng: {row['id']}")
            all_ids.add(row["id"])
            for key in content_keys(row):
                if owners.setdefault(key, name) != name:
                    raise ValueError(f"Nội dung rò rỉ giữa các tập: {row['id']}")
        splits[name] = rows
    membership = {row["id"]: name for name, rows in splits.items() for row in rows}
    for left, right in near_pairs((row for rows in splits.values() for row in rows), text_key):
        if membership[left] != membership[right]:
            raise ValueError(f"Cặp câu gần trùng giữa các tập: {left}, {right}")
    return splits, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=PROCESSED)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    splits, _ = validate(args.directory, args.manifest)
    print("Đã xác minh phân chia VSEC:", {key: len(value) for key, value in splits.items()})


if __name__ == "__main__":
    main()
