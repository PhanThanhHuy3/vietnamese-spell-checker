"""Fine-tune a pretrained Transformer on the audited VSEC train split."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import random
from uuid import uuid4
from time import perf_counter

from generate_noise_data import augment
from src.data.prepare_vsec import write_json
from src.data.validate_split import validate
from src.utils.helpers import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--noise-size", type=int, default=0)
    parser.add_argument("--noise-prob", type=float, default=.2)
    parser.add_argument("--identity-size", type=int)
    parser.add_argument("--identities", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    settings, model = config["training"], config["model"]
    for key, value in {
        "num_train_epochs": args.epochs, "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size, "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
    }.items():
        if value is not None:
            if value <= 0:
                parser.error(f"{key} phải lớn hơn 0")
            settings[key] = value
    if args.smoke_steps is not None and (args.smoke_steps < 1 or args.output is None):
        parser.error("--smoke-steps cần số dương và thư mục --output riêng")
    output = args.output or Path(settings["output_dir"])
    if output.exists() and any(output.iterdir()):
        parser.error(f"Thư mục {output} đã có dữ liệu. Chọn thư mục thử nghiệm mới")
    settings["output_dir"] = str(output)
    splits, manifest = validate()
    training_rows = splits["train"]
    extras = augment(splits, args.noise_size, args.noise_prob, config["seed"], args.identities, args.identity_size)
    val_rows = splits["val"]
    if args.smoke_steps:
        rng = random.Random(config["seed"])
        training_rows = rng.sample(training_rows, min(32, len(training_rows)))
        val_rows = rng.sample(val_rows, min(8, len(val_rows)))

    from src.data.dataset import SpellingDataset
    from src.models.spell_corrector import VietnameseSpellCorrector
    from src.training.trainer import SpellingTrainer
    from src.utils.helpers import set_seed

    pretrained = {"vit5": "VietAI/vit5-base", "byt5": "google/byt5-small"}
    if pretrained.get(model["key"]) != model["name_or_path"]:
        parser.error("Huấn luyện bắt đầu từ mô hình pretrained gốc đã định danh")
    set_seed(config["seed"])
    wrapper = VietnameseSpellCorrector(model["name_or_path"], local_only=args.local_only)
    limits = (model["max_source_length"], model["max_target_length"], model["prefix"])
    train = SpellingDataset(training_rows + extras, wrapper.tokenizer, *limits)
    val = SpellingDataset(val_rows, wrapper.tokenizer, *limits)
    print(f"Train {len(train)}, validation {len(val)}. Vượt giới hạn token: "
          f"train {len(train.excluded_ids)}, validation {len(val.excluded_ids)}", flush=True)
    trainer = SpellingTrainer(wrapper, train, val, config, args.smoke_steps)
    # Keep the model unpublished until the complete training run has written its manifest.
    started = perf_counter()
    trainable = sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad)
    trainer.train()
    state = trainer.runner.state
    write_json(output / "training_history.json", state.log_history)
    checkpoint = output / "best_model"
    wrapper.save_pretrained(checkpoint)
    metadata = {
        "training_id": str(uuid4()), "completed": True, "smoke": args.smoke_steps is not None,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": manifest["raw_sha256"],
        "splits": {key: entry["sha256"] for key, entry in manifest["splits"].items()},
        "model_key": model["key"], "model": model, "generation": config["generation"],
        "training": settings, "seed": config["seed"], "noise_size": args.noise_size,
        "augmentation_counts": dict(Counter(row["noise_type"] for row in extras)),
        "noise_probability": args.noise_prob, "identities": sum(row["noise_type"] == "identity" for row in extras),
        "parameters": sum(parameter.numel() for parameter in wrapper.model.parameters()),
        "pretrained_revision": wrapper.model.config._commit_hash,
        "trainable_parameters": trainable, "epochs_completed": state.epoch,
        "global_steps": state.global_step, "train_runtime_s": round(perf_counter() - started, 3),
        "device": str(wrapper.device), "training_samples": len(train), "validation_samples": len(val),
        "excluded_ids": {"train": train.excluded_ids, "val": val.excluded_ids},
        "history": state.log_history,
    }
    write_json(checkpoint / "training_meta.json", metadata)
    write_json(output / "model_manifest.json", metadata)


if __name__ == "__main__":
    main()
