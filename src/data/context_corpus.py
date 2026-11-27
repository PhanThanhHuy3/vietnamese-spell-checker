"""Read the vendored Vietnamese Tatoeba export with holdout overlap removed."""
import bz2
import csv
from pathlib import Path

from src.data.prepare_vsec import ROOT, content_keys, read_rows, text_key
from src.postprocessing import without_accents

CONTEXT_PATH = ROOT / "data/context/vie_sentences_detailed.tsv.bz2"


def context_sentences(splits, root=ROOT):
    path = root / CONTEXT_PATH.relative_to(ROOT)
    if not path.is_file():
        return
    excluded = {without_accents(key) for name in ("train", "val", "test")
                for row in splits[name] for key in content_keys(row)}
    for name in ("practice", "challenge"):
        examples = root / "data" / f"{name}.jsonl"
        if examples.is_file():
            for row in read_rows(examples):
                alternatives = row.get("acceptable_targets", ())
                excluded.update(without_accents(text_key(term))
                                for term in (row["input_text"], row["target_text"], *alternatives))

    # The official export stores sentence ID, language, text, author and dates.
    with bz2.open(path, "rt", encoding="utf-8", newline="") as handle:
        for sentence_id, language, text, *_ in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            key = without_accents(text_key(text))
            if language == "vie" and sentence_id.isdecimal() and key and key not in excluded:
                excluded.add(key)
                yield text
