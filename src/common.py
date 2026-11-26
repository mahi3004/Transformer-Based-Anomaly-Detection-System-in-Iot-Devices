# src/common.py
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELS = {"benign": 0, "mirai": 1, "gafgyt": 2}
CLASS_NAMES = ["Benign", "Mirai", "Gafgyt"]


def parse_name(path):
    parts = Path(path).stem.split(".")
    device = int(parts[0])
    class_name = parts[1].lower()
    return device, class_name


def get_traffic_files():
    return sorted(
        path for path in DATA_DIR.rglob("*.csv")
        if path.stem.split(".")[0].isdigit()
    )
