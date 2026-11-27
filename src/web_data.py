"""Read-only dataset exploration and reproducible experiment artifact access."""
from collections import Counter, defaultdict
from functools import cached_property
from itertools import combinations
import json
import random
import re

from rapidfuzz.distance import Levenshtein

from src.data.prepare_vsec import content_keys, file_hash, read_rows, text_key
from src.data.preprocess import normalize_vietnamese_text
from src.data.validate_split import validate
from src.postprocessing import WORD, error_category, without_accents

VSEC_SOURCE = {
    "name": "VSEC Corpus", "authors": "VSEC2021",
    "url": "https://github.com/VSEC2021/VSEC",
    "version": "SHA-256 của bản JSONL trong manifest",
    "terms": "Xem điều khoản và thông tin ghi công trong kho phát hành.",
}
DATASET_SPLITS = ("train", "val", "test")
EVALUATION_SPLITS = ("test", "val", "practice", "challenge", "clean_test")
RUN_NAME = re.compile(r"[A-Za-z0-9_.-]{1,80}")


def annotation_type(before, after):
    a, b = before.casefold(), after.casefold()
    if not a or not b or len(a.split()) != len(b.split()):
        return "Tách hoặc ghép âm tiết"
    if without_accents(a) == without_accents(b):
        return "Dấu tiếng Việt"
    if len(a) != len(b):
        return "Thừa hoặc thiếu ký tự"
    return error_category(before, after)


class WebsiteData:
    def __init__(self, root, service):
        self.root = root
        self.service = service

    @cached_property
    def raw_rows(self):
        path = self.root / "data/raw/VSEC.jsonl"
        if not path.is_file() or file_hash(path) != self.service.manifest["raw_sha256"]:
            return {}
        records = {}
        with path.open(encoding="utf-8-sig") as handle:
            for index, line in enumerate(handle, 1):
                try:
                    records[f"vsec_{index:05d}"] = json.loads(line)
                except ValueError:
                    continue
        return records

    @cached_property
    def by_type(self):
        result = {}
        for identifier, raw in self.raw_rows.items():
            result[identifier] = Counter(
                annotation_type(item["current_syllable"], item["alternative_syllables"][0])
                for item in raw["annotations"] if not item["is_correct"] and item["alternative_syllables"]
            )
        return result

    @cached_property
    def stats(self):
        manifest = self.service.manifest
        splits = {}
        for name, rows in self.service.splits.items():
            lengths = Counter(len(WORD.findall(row["input_text"])) for row in rows)
            types = Counter()
            for row in rows:
                types.update(self.by_type.get(row["id"], {}))
            syllables = sum(length * count for length, count in lengths.items())
            splits[name] = {**{k: v for k, v in manifest["splits"][name].items() if k != "ids"},
                            "syllables": syllables, "mean_length": syllables / len(rows),
                            "length_histogram": dict(sorted(lengths.items())), "error_types": dict(types)}
        path = self.root / "data/synthetic/train.jsonl"
        augmentation = Counter(row.get("noise_type", "unknown") for row in read_rows(path)) if path.is_file() else {}
        return {"source": VSEC_SOURCE, "raw_sha256": manifest["raw_sha256"], "seed": manifest["seed"],
                "duplicates": manifest["duplicates"], "rejected_lines": len(manifest["rejected_lines"]),
                "raw_annotations_verified": bool(self.raw_rows), "augmentation": dict(augmentation), "splits": splits}

    def samples(self, filters):
        name = filters.get("split", "train")
        if name not in DATASET_SPLITS:
            raise ValueError("Tập dữ liệu không hợp lệ.")
        term = filters.get("q", "").casefold()
        kind = filters.get("type", "")
        selected_id = filters.get("id")
        rows = [row for row in self.service.splits[name]
                if (not term or any(term in row[key].casefold() for key in ("id", "input_text", "target_text")))
                and (not kind or kind in self.by_type.get(row["id"], {}))
                and (not selected_id or row["id"] == selected_id)]
        if filters.get("random") == "1" and rows:
            rows = [random.choice(rows)]
        limit = max(1, min(100, int(filters.get("limit", 12))))
        result = []
        for row in rows[:limit]:
            item = {**row, "types": list(self.by_type.get(row["id"], {}))}
            if selected_id or filters.get("random") == "1":
                raw = self.raw_rows.get(row["id"], {})
                item.update(annotation=raw.get("annotations"), raw_text=raw.get("text"),
                            normalized_text=normalize_vietnamese_text(raw.get("text", "")))
            if filters.get("tokenize") == "1" and selected_id:
                item["tokenization"] = self.tokenization(row)
            result.append(item)
        return {"split": name, "matched": len(rows), "samples": result}

    @cached_property
    def demo_tokenizer(self):
        from transformers import AutoTokenizer
        path = self.root / "checkpoints/vit5_spelling_corrector/best_model"
        source = str(path) if path.is_dir() else "VietAI/vit5-base"
        return AutoTokenizer.from_pretrained(source, local_files_only=True)

    def tokenization(self, row):
        try:
            tokenizer = self.demo_tokenizer
            groups = {}
            for field, text in (("input", "spelling: " + row["input_text"]), ("target", row["target_text"])):
                ids = tokenizer.encode(text)
                groups[field] = {"tokens": tokenizer.convert_ids_to_tokens(ids), "ids": ids}
            return {"tokenizer": "ViT5 SentencePiece", **groups}
        except (OSError, ImportError, ValueError) as exc:
            return {"error": f"Chưa nạp được tokenizer ViT5: {exc}"}

    def integrity(self):
        splits = self.service.splits
        try:
            splits, _ = validate(self.root / "data/processed", self.root / "data/splits/split_manifest.json")
            manifest_ok, problem = True, None
        except (OSError, ValueError, KeyError) as exc:
            manifest_ok, problem = False, str(exc)
        checks = {}
        for left, right in combinations(DATASET_SPLITS, 2):
            a, b = splits[left], splits[right]
            overlap = lambda x, y: len(x & y)
            fields = {field: overlap({text_key(row[field]) for row in a}, {text_key(row[field]) for row in b})
                      for field in ("source_text", "input_text", "target_text")}
            checks[f"{left}/{right}"] = {
                "id": overlap({row["id"] for row in a}, {row["id"] for row in b}), **fields,
                "cross_field": overlap(set().union(*(content_keys(row) for row in a)),
                                       set().union(*(content_keys(row) for row in b))),
            }
        train = {row["id"]: row for row in splits["train"]}
        held_out = {without_accents(k) for name in ("val", "test") for row in splits[name] for k in content_keys(row)}
        path = self.root / "data/synthetic/train.jsonl"
        noise = read_rows(path) if path.is_file() else []
        invalid = sum(row.get("parent_id") not in train
                      or row.get("target_text") != train[row["parent_id"]]["target_text"]
                      or any(without_accents(k) in held_out for k in content_keys(row))
                      for row in noise)
        near = self.near_matches(splits)
        return {"manifest_ok": manifest_ok, "problem": problem, "checks": checks, "near_matches": near,
                "source_documents": "VSEC JSONL không có ID tài liệu gốc để đối chiếu.",
                "augmentation": {"checked": len(noise), "invalid": invalid if noise else None},
                "baseline_vsec_source": "Train. Ngữ liệu bổ sung được lọc theo khóa nội dung VSEC."}

    def near_matches(self, splits):
        # Compare sentences that share a 5-token edge. Exact overlap is checked separately.
        index = defaultdict(list)
        findings = set()
        for name, rows in splits.items():
            for row in rows:
                text = text_key(row["target_text"])
                words = text.split()
                if len(words) < 8:
                    continue
                edges = {tuple(words[:5]), tuple(words[-5:])}
                for edge in edges:
                    for old_name, old_id, previous in index[edge]:
                        if old_name != name and text != previous and Levenshtein.normalized_similarity(text, previous) >= .9:
                            findings.add((old_id, row["id"]))
                    index[edge].append((name, row["id"], text))
        return {"method": "Chung 5 âm tiết đầu hoặc cuối, tương đồng ký tự từ 90%",
                "pairs": len(findings), "examples": sorted(findings)[:10]}

    def experiments(self):
        train_hash = self.service.manifest["splits"]["train"]["sha256"]
        result = []
        for split in EVALUATION_SPLITS:
            if split in ("test", "val"):
                digest = self.service.manifest["splits"][split]["sha256"]
            else:
                path = self.root / "data" / ("evaluation/clean_test.jsonl" if split == "clean_test" else f"{split}.jsonl")
                if not path.is_file():
                    continue
                digest = file_hash(path)
            for path in sorted((self.root / "experiments/results" / split).glob("*.json")):
                try:
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if (row["split_sha256"] == digest and not row["smoke"] and row["train_sha256"] == train_hash):
                        result.append({"id": f"{split}--{path.stem}", **row})
                except (ValueError, OSError, KeyError, TypeError):
                    continue
        return result

    def experiment(self, identifier, section=None, filters=None):
        entry = next((row for row in self.experiments() if row["id"] == identifier), None)
        if entry is None:
            return None
        if section == "history":
            return {"history": entry.get("history", [])}
        if section == "predictions":
            split, run = identifier.split("--", 1)
            if not RUN_NAME.fullmatch(run):
                return None
            path = self.root / "experiments/results" / split / f"{run}.predictions.jsonl"
            rows = read_rows(path) if path.is_file() else []
            kind = (filters or {}).get("error", "")
            if kind:
                rows = [row for row in rows if (row.get("errors") or {}).get(kind, 0)]
            return {"matched": len(rows), "samples": rows[:100]}
        return entry
