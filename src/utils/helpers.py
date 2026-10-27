import os
import random
from pathlib import Path
from typing import Any, Dict


def set_seed(seed: int = 42) -> None:
    """Cố định seed để đảm bảo tính tái lập (reproducibility) của thực nghiệm."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Đọc file cấu hình YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {path}")

    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except ImportError:
        raise ImportError(
            "Chưa cài đặt thư viện 'pyyaml'. Vui lòng chạy: pip install pyyaml hoặc pip install -r requirements.txt"
        )


def get_device():
    """Tự động kiểm tra và trả về thiết bị tính toán (CUDA / MPS / CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        raise ImportError(
            "Chưa cài đặt thư viện 'torch'. Vui lòng chạy: pip install torch hoặc pip install -r requirements.txt"
        )
