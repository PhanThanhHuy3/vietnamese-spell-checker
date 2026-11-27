"""Run the VSEC experiment matrix through the same training CLI."""
import argparse
from pathlib import Path
import subprocess
import sys

RUNS = {
    "exp02": (1, 0, .2, 1e-4, 4, 4),
    "exp03": (3, 0, .2, 1e-4, 4, 4),
    "exp04": (3, 5000, .2, 1e-4, 4, 4),
    "exp05": (5, 10000, .2, 1e-4, 4, 4),
    "exp06": (3, 5000, .2, 1e-4, 4, 4),
    "noise01": (3, 5000, .1, 1e-4, 4, 4),
    "noise03": (3, 5000, .3, 1e-4, 4, 4),
    "noise20k": (3, 20000, .2, 1e-4, 4, 4),
    "lr_high": (3, 5000, .2, 2e-4, 4, 4),
    "lr_low": (3, 5000, .2, 5e-5, 4, 4),
    "batch8": (3, 5000, .2, 1e-4, 8, 4),
}


def run(script, *args):
    subprocess.run([sys.executable, "-X", "utf8", script, *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", choices=["exp01", "exp07", *RUNS])
    parser.add_argument("--candidate", type=Path, nargs="+")
    args = parser.parse_args()
    if "exp07" in args.runs and not args.candidate:
        parser.error("EXP07 cần --candidate trỏ tới một hay nhiều checkpoint ViT5")
    for name in args.runs:
        if name == "exp01":
            run("evaluate.py", "--model", "baseline")
            continue
        if name == "exp07":
            run("select_checkpoint.py", "--model", "vit5", "--checkpoints", *args.candidate)
            run("evaluate.py", "--model", "vit5", "--run_name", name)
            continue
        epochs, size, probability, lr, batch, accumulation = RUNS[name]
        model = "byt5" if name == "exp06" else "vit5"
        output = Path("experiments") / name
        config = "configs/byt5.yaml" if model == "byt5" else "configs/config.yaml"
        run("train.py", "--config", config, "--output", output, "--epochs", epochs,
            "--noise-size", size, "--noise-prob", probability, "--lr", lr,
            "--batch_size", batch, "--gradient-accumulation", accumulation,
            *(["--identities"] if size else []))
        run("evaluate.py", "--model", model, "--model_path", output / "best_model",
            "--split", "val", "--run_name", name)


if __name__ == "__main__":
    main()
