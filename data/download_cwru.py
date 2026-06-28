"""
CWRU Bearing Dataset Downloader
================================
Automatically downloads the Case Western Reserve University (CWRU)
bearing fault dataset.

Reference: https://engineering.case.edu/bearingdatacenter

The dataset contains:
- Normal baseline data (4 loads: 0, 1, 2, 3 hp)
- Inner race fault (0.007", 0.014", 0.021" diameters)
- Outer race fault (0.007", 0.014", 0.021" diameters)
- Ball fault (0.007", 0.014", 0.021" diameters)
- Sampling rate: 12 kHz (drive-end) / 48 kHz (fan-end)

Usage:
    python download_cwru.py                     # download all
    python download_cwru.py --load 0 1 2 3      # specific loads only
    python download_cwru.py --fault_type IR     # specific fault type
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# ========== CWRU File URLs ==========
# CWRU data is publicly available via the Case website.
# Each .mat file corresponds to one (fault_type, fault_size, load) combination.

_BASE_URL = "https://engineering.case.edu/bearingdatacenter/sites/bearingdatacenter.case.edu/files/uploads/"

# File naming: <fault_type>_<fault_size>_<load>.mat
# Fault types: Normal, IR (Inner Race), OR (Outer Race), B (Ball)
# Fault sizes: 007, 014, 021 (inches x 1000)
# Loads: 0, 1, 2, 3 (hp motor load)

CWRU_FILES = {
    # (fault_type, fault_size, load_hp) -> filename
    # --- Normal ---
    ("Normal", None, 0):  "97.mat",
    ("Normal", None, 1):  "98.mat",
    ("Normal", None, 2):  "99.mat",
    ("Normal", None, 3):  "100.mat",
    # --- Inner Race 0.007" ---
    ("IR", "007", 0): "105.mat",
    ("IR", "007", 1): "106.mat",
    ("IR", "007", 2): "107.mat",
    ("IR", "007", 3): "108.mat",
    # --- Inner Race 0.014" ---
    ("IR", "014", 0): "169.mat",
    ("IR", "014", 1): "170.mat",
    ("IR", "014", 2): "171.mat",
    ("IR", "014", 3): "172.mat",
    # --- Inner Race 0.021" ---
    ("IR", "021", 0): "209.mat",
    ("IR", "021", 1): "210.mat",
    ("IR", "021", 2): "211.mat",
    ("IR", "021", 3): "212.mat",
    # --- Outer Race 0.007" ---
    ("OR", "007", 0): "130.mat",
    ("OR", "007", 1): "131.mat",
    ("OR", "007", 2): "132.mat",
    ("OR", "007", 3): "133.mat",
    # --- Outer Race 0.014" ---
    ("OR", "014", 0): "197.mat",
    ("OR", "014", 1): "198.mat",
    ("OR", "014", 2): "199.mat",
    ("OR", "014", 3): "200.mat",
    # --- Outer Race 0.021" ---
    ("OR", "021", 0): "234.mat",
    ("OR", "021", 1): "235.mat",
    ("OR", "021", 2): "236.mat",
    ("OR", "021", 3): "237.mat",
    # --- Ball 0.007" ---
    ("B", "007", 0):  "118.mat",
    ("B", "007", 1):  "119.mat",
    ("B", "007", 2):  "120.mat",
    ("B", "007", 3):  "121.mat",
    # --- Ball 0.014" ---
    ("B", "014", 0):  "185.mat",
    ("B", "014", 1):  "186.mat",
    ("B", "014", 2):  "187.mat",
    ("B", "014", 3):  "188.mat",
    # --- Ball 0.021" ---
    ("B", "021", 0):  "222.mat",
    ("B", "021", 1):  "223.mat",
    ("B", "021", 2):  "224.mat",
    ("B", "021", 3):  "225.mat",
}

# Drive-end accelerometer data (DE), sampling rate 12 kHz
# Each .mat file has variable containing the signal:
#   - DE:  drive-end vibration signal
#   - FE:  fan-end vibration signal (48 kHz)
#   - BA:  base accelerometer

# Label mapping: fault_type + fault_size -> class_id
_LABEL_MAP = {
    "Normal": 0,
    "IR007": 1, "IR014": 2, "IR021": 3,
    "OR007": 4, "OR014": 5, "OR021": 6,
    "B007": 7,  "B014": 8,  "B021": 9,
}

NUM_CLASSES = len(_LABEL_MAP)  # 10


def get_label(fault_type: str, fault_size: str = None) -> int:
    """Map fault type + size to integer class label."""
    if fault_type == "Normal":
        return _LABEL_MAP["Normal"]
    key = f"{fault_type}{fault_size}"
    return _LABEL_MAP[key]


def download_cwru(
    save_dir: str = "data/raw/CWRU",
    loads: list = None,
    fault_types: list = None,
) -> None:
    """
    Download CWRU .mat files.

    Args:
        save_dir:  Directory to save files
        loads:     Motor loads to download, default [0, 1, 2, 3]
        fault_types: Fault types, default ['Normal', 'IR', 'OR', 'B']
    """
    if loads is None:
        loads = [0, 1, 2, 3]
    if fault_types is None:
        fault_types = ["Normal", "IR", "OR", "B"]

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    total = 0
    downloaded = 0
    skipped = 0

    for (ft, fs, ld), filename in CWRU_FILES.items():
        if ft not in fault_types or ld not in loads:
            continue
        total += 1
        filepath = save_path / filename
        if filepath.exists():
            skipped += 1
            continue

        url = _BASE_URL + filename
        try:
            print(f"  Downloading {filename} ({ft}@{ld}hp)...", end=" ")
            urllib.request.urlretrieve(url, filepath)
            print("OK")
            downloaded += 1
        except urllib.error.URLError as e:
            print(f"FAILED: {e}")
            print(f"  → CWRU website may have changed URLs.")
            print(f"  → Please manually download from:")
            print(f"    https://engineering.case.edu/bearingdatacenter/download-data-file")
            sys.exit(1)

    print(f"\nDone: {downloaded} downloaded, {skipped} already exist, {total} total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CWRU bearing dataset")
    parser.add_argument("--save_dir", type=str, default="data/raw/CWRU")
    parser.add_argument("--load", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--fault_type", type=str, nargs="+",
                        default=["Normal", "IR", "OR", "B"])
    args = parser.parse_args()

    print("=" * 60)
    print("CWRU Bearing Dataset Downloader")
    print("=" * 60)
    print(f"Save dir: {args.save_dir}")
    print(f"Loads: {args.load}")
    print(f"Fault types: {args.fault_type}")
    print()

    download_cwru(args.save_dir, args.load, args.fault_type)
