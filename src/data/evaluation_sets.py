"""Create evaluation-only controls from held-out VSEC targets."""
from src.data.prepare_vsec import ROOT, content_keys, write_rows
from src.postprocessing import without_accents


def clean_test(splits, root=ROOT):
    train = {without_accents(key) for row in splits["train"] for key in content_keys(row)}
    rows = []
    for row in splits["test"]:
        text = row["target_text"]
        if any(without_accents(key) in train for key in content_keys(row)):
            continue
        rows.append({"id": "clean_" + row["id"], "parent_id": row["id"], "source_split": "test",
                     "input_text": text, "target_text": text})
    path = root / "data/evaluation/clean_test.jsonl"
    write_rows(path, rows)
    return path
