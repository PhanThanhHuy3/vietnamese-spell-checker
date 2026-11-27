"""Run the experiments defined for VietSpell AI."""
import argparse
from pathlib import Path
import subprocess
import sys

RUNS = {
    "exp02": ("vit5", 0, 0, 3, 1e-4),
    "exp03": ("vit5", 5000, 0, 3, 1e-4),
    "exp04": ("vit5", 5000, 1868, 3, 1e-4),
    "exp05": ("vit5", 5000, 1868, 4, 5e-5),
    "exp07": ("byt5", 5000, 1868, 3, 1e-4),
}


def run(script, *args):
    subprocess.run([sys.executable, "-X", "utf8", script, *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", choices=("exp00", "exp01", *RUNS, "exp06"))
    parser.add_argument("--candidate", type=Path, nargs="+")
    args = parser.parse_args()
    if "exp06" in args.runs and not args.candidate:
        parser.error("EXP06 cần --candidate trỏ tới các checkpoint ViT5 đã huấn luyện.")
    for name in args.runs:
        if name in ("exp00", "exp01"):
            model = "identity" if name == "exp00" else "baseline"
            run("evaluate.py", "--model", model, "--split", "test", "--run_name", name)
            if model == "baseline":
                run("evaluate.py", "--model", model, "--split", "clean_test", "--run_name", name)
            continue
        if name == "exp06":
            run("select_checkpoint.py", "--model", "vit5", "--checkpoints", *args.candidate)
            run("evaluate.py", "--model", "vit5", "--split", "val", "--run_name", name)
            continue
        model, noise, identities, epochs, lr = RUNS[name]
        output = Path("experiments") / name
        config = "configs/byt5.yaml" if model == "byt5" else "configs/config.yaml"
        run("train.py", "--config", config, "--output", output, "--epochs", epochs,
            "--noise-size", noise, "--identity-size", identities, "--lr", lr)
        run("evaluate.py", "--model", model, "--model_path", output / "best_model",
            "--split", "val", "--run_name", name)


if __name__ == "__main__":
    main()
