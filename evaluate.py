"""Evaluate a train-only baseline or a verified checkpoint on the same VSEC split."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from src.data.noise_synthesizer import VietnameseNoiseSynthesizer
from src.data.prepare_vsec import ROOT, content_keys, file_hash, read_rows, text_key, write_json, write_rows
from src.data.validate_split import validate
from src.inference import MODEL_PATHS, SpellInferencePipeline, checkpoint_metadata
from src.data.context_corpus import context_sentences
from src.models.baseline import NgramBaseline
from src.training.metrics import summarize
from src.postprocessing import apply_edits, without_accents


def stress_metrics(predict, targets, train_rows):
    forbidden = {without_accents(key) for row in train_rows for key in content_keys(row)}
    generator = VietnameseNoiseSynthesizer(seed=42)
    results = {}
    for mode in ("full_diacritics", "partial_diacritics", "mixed"):
        sources, gold = [], []
        for text in targets:
            variant = generator.accent_variant(text, .35, mode)
            if variant != text and without_accents(text_key(variant)) not in forbidden:
                sources.append(variant)
                gold.append(text)
        if sources:
            metrics = summarize(predict(sources), sources, gold)
            metrics.pop("failures")
            results[mode] = metrics
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", *MODEL_PATHS), default="baseline")
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--split", choices=("val", "test", "practice", "challenge"), default="test")
    parser.add_argument("--run_name")
    parser.add_argument("--unrestricted-distance", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--stress", action="store_true", help="Đo riêng các nhóm lỗi trên biến thể của tập đánh giá")
    args = parser.parse_args()
    run = args.run_name or args.model
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", run) or run in (".", ".."):
        parser.error("Tên lần chạy không hợp lệ")
    splits, manifest = validate()
    if args.split in ("practice", "challenge"):
        path = ROOT / "data" / f"{args.split}.jsonl"
        rows, digest = read_rows(path), file_hash(path)
    else:
        rows, digest = splits[args.split], manifest["splits"][args.split]["sha256"]
    source = [row["input_text"] for row in rows]
    target = [row["target_text"] for row in rows]
    meta = None
    if args.model == "baseline":
        engine = NgramBaseline(splits["train"], context_sentences(splits))
        predict = lambda texts: [engine.correct(text) for text in texts]
    else:
        path = args.model_path or ROOT / MODEL_PATHS[args.model]
        meta = checkpoint_metadata(path, manifest)
        if meta["model_key"] != args.model or (meta.get("smoke") and not args.allow_smoke):
            parser.error("Checkpoint không phù hợp với phép đánh giá")
        engine = SpellInferencePipeline(path, meta, conservative=not args.unrestricted_distance)
        predict = engine.correct_many
    def predict_batches(texts):
        results = []
        for index in range(0, len(texts), 16):
            results.extend(predict(texts[index:index + 16]))
        return results

    predictions = predict_batches(source)
    if args.split == "challenge":
        accepted_targets = [(row["target_text"], *row.get("acceptable_targets", ())) for row in rows]
        scoring_targets = [prediction if prediction in choices else choices[0]
                           for prediction, choices in zip(predictions, accepted_targets, strict=True)]
        metrics = summarize(predictions, source, scoring_targets)
        groups = {}
        for index, row in enumerate(rows):
            phase = groups.setdefault(row["phase"], {"samples": 0, "exact": 0, "clean": 0, "clean_preserved": 0})
            phase["samples"] += 1
            phase["exact"] += predictions[index] in accepted_targets[index]
            if row["input_text"] == row["target_text"]:
                phase["clean"] += 1
                phase["clean_preserved"] += predictions[index] == row["input_text"]
        metrics["challenge_phases"] = groups
        if args.model == "baseline":
            auto_predictions = [apply_edits(text, [edit for edit in engine.suggest(text) if edit["auto_apply"]])
                                for text in source]
            metrics["automatic_exact"] = sum(p in choices for p, choices in zip(auto_predictions, accepted_targets, strict=True))
        else:
            auto_predictions = predictions
    else:
        metrics = summarize(predictions, source, target)
        auto_predictions = None
    clean_predictions = predict_batches(target)
    clean_modified = sum(p != t for p, t in zip(clean_predictions, target, strict=True))
    metrics["clean_reference_control"] = {
        "samples": len(target), "modified": clean_modified,
        "rate": clean_modified / len(target),
    }
    if args.stress:
        metrics["noise_groups"] = stress_metrics(predict_batches, target, splits["train"])
    failures = {entry["index"]: entry for entry in metrics.pop("failures")}
    output = ROOT / "experiments" / "results" / args.split
    write_rows(output / f"{run}.predictions.jsonl", [
        dict(id=row["id"], input_text=row["input_text"], target_text=row["target_text"],
             prediction=prediction, errors=failures.get(index),
             **({"automatic_prediction": auto_predictions[index], "phase": row["phase"]}
                if auto_predictions is not None else {}))
        for index, (row, prediction) in enumerate(zip(rows, predictions, strict=True))
    ])
    result = dict(
        run=run, model=args.model, split=args.split, split_sha256=digest,
        train_sha256=manifest["splits"]["train"]["sha256"], metrics=metrics,
        smoke=bool(meta and meta.get("smoke")), training_id=meta["training_id"] if meta else None,
        generation=(meta["generation"] if meta else None), conservative=not args.unrestricted_distance,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        history=meta["history"] if meta else [], error_examples=[
            dict(id=rows[i]["id"], source=source[i], reference=target[i], prediction=predictions[i], **failure)
            for i, failure in list(failures.items())[:10]
        ],
    )
    write_json(output / f"{run}.json", result)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
