"""Tune decoding and select a deployment checkpoint on VSEC validation only."""
import argparse
from pathlib import Path
import shutil
import tempfile

from src.data.prepare_vsec import ROOT, write_json
from src.data.validate_split import validate
from src.inference import MODEL_PATHS, SpellInferencePipeline, checkpoint_metadata
from src.postprocessing import apply_edits
from src.training.metrics import evaluate_edits


def find_best(paths, manifest, validation, model, beams_options):
    sources = [row["input_text"] for row in validation]
    targets = [row["target_text"] for row in validation]
    texts = sources + targets
    best = None
    table = []
    for path in paths:
        meta = checkpoint_metadata(path, manifest)
        if meta["model_key"] != model or meta.get("smoke"):
            raise ValueError(f"Checkpoint không phù hợp để triển khai: {path}")
        wrapper = SpellInferencePipeline(path, meta)
        for beams in beams_options:
            wrapper.beams = beams
            chunks = wrapper.scored_chunks(texts)
            scores = sorted(score for _, _, _, _, score in chunks if score is not None)
            thresholds = [None] + sorted({scores[int((len(scores) - 1) * q)]
                                         for q in (.1, .25, .5, .75) if scores})
            for threshold in thresholds:
                edits = wrapper.filter_chunks(texts, chunks, threshold)
                predictions = [apply_edits(text, changes) for text, changes in zip(texts, edits, strict=True)]
                correction = evaluate_edits(predictions[:len(sources)], sources, targets)["correction"]
                clean_rate = sum(p != t for p, t in zip(predictions[len(sources):], targets, strict=True)) / len(targets)
                tp, fp, fn = (correction[key] for key in ("true_positive", "false_positive", "false_negative"))
                f_half = 1.25 * tp / max(1.25 * tp + fp + .25 * fn, 1)
                objective = f_half - .25 * clean_rate
                entry = dict(training_id=meta["training_id"], beams=beams, min_log_probability=threshold,
                             correction=correction, clean_modified_rate=clean_rate, objective=objective)
                table.append(entry)
                ranking = (correction["f1"], -clean_rate, objective, -beams)
                if best is None or ranking > best[0]:
                    best = ranking, path, meta, entry
        del wrapper
    return best, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_PATHS, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--beams", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    if any(beam < 1 or beam > 8 for beam in args.beams):
        parser.error("Beam size phải từ 1 đến 8")
    destination = args.output or ROOT / MODEL_PATHS[args.model]
    if destination.exists():
        parser.error(f"Đích đã tồn tại: {destination}. Chọn --output mới")
    splits, manifest = validate()
    (_, source, meta, entry), candidates = find_best(
        args.checkpoints, manifest, splits["val"], args.model, args.beams
    )
    meta["generation"].update(num_beams=entry["beams"], min_log_probability=entry["min_log_probability"])
    meta["calibration"] = {"selected": entry, "candidates": candidates}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / "best_model"
        shutil.copytree(source, staged)
        write_json(staged / "training_meta.json", meta)
        staged.rename(destination)
    print(destination)


if __name__ == "__main__":
    main()
