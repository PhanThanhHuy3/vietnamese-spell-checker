"""Resolve the offline pretrained snapshots used for fine-tuning."""
from hashlib import sha256
from pathlib import Path

VIT5_MODEL = "VietAI/vit5-base"
BYT5_MODEL = "google/byt5-small"
SNAPSHOTS = {
    VIT5_MODEL: (
        "2209a38d735ede63e88f5aa52bcdc11a05a37b85",
        ("tokenizer.json",),
        {
            "pytorch_model.bin": "6e666460219d3495e6437ac09061425ccf896532c92ec00cd3e6130cc94f3f44",
            "spiece.model": "59986b62f9f0b90edafb9b073ea7b93d21114a5841219a1ea2399ade73f729c6",
        },
    ),
    BYT5_MODEL: (
        "68377bdc18a2ffec8a0533fef03b1c513a4dd49d",
        ("special_tokens_map.json",),
        {"pytorch_model.bin": "5c5aaf56299d6f2d4eaadad550a40765198828ead4d74f0a15f91cbe0961931a"},
    ),
}


def resolve_pretrained(model):
    """Use the local snapshot when present, with the Hub ID as the portable default."""
    identifier = model["name_or_path"]
    directory = model.get("pretrained_dir")
    if not directory:
        return identifier, None
    path = Path(directory).resolve()
    if not path.exists():
        return identifier, None
    if identifier not in SNAPSHOTS or not path.is_dir():
        raise ValueError(f"Thư mục pretrained không hợp lệ: {path}")
    if (path / "model.safetensors").exists() or (path / "model.safetensors.index.json").exists():
        raise ValueError(f"Thư mục {identifier} pretrained phải dùng đúng trọng số gốc pytorch_model.bin.")
    revision, tokenizer_files, checksums = SNAPSHOTS[identifier]
    required = ("config.json", "tokenizer_config.json", *tokenizer_files, *checksums)
    for name in required:
        if not (path / name).is_file():
            raise ValueError(f"Thiếu {name} trong thư mục pretrained {path}")
    for name, expected in checksums.items():
        digest = sha256()
        with (path / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"{name} không khớp bản {identifier} pretrained đã xác minh.")
    return str(path), revision
