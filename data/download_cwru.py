"""
CWRU Bearing Dataset Downloader
================================
Downloads the Case Western Reserve University (CWRU) bearing fault dataset
via a GitHub mirror (original Case website URLs are no longer stable).

Usage:
    python download_cwru.py                     # clone from GitHub mirror
"""

import subprocess
import sys
from pathlib import Path

_MIRROR_REPO = "https://github.com/AiChiXiaoXiongBingGan/CWRU-dataset.git"

# Fault type + size → subdirectory name in the mirror repo
_FAULT_DIR_MAP = {
    ("Normal", None):  "normal",
    ("IR", "007"):     "inner_07",
    ("IR", "014"):     "inner_14",
    ("IR", "021"):     "inner_21",
    ("OR", "007"):     "outer_07",
    ("OR", "014"):     "outer_14",
    ("OR", "021"):     "outer_21",
    ("B", "007"):      "ball_07",
    ("B", "014"):       "ball_14",
    ("B", "021"):       "ball_21",
}

# Label mapping
_LABEL_MAP = {
    "Normal": 0,
    "IR007": 1, "IR014": 2, "IR021": 3,
    "OR007": 4, "OR014": 5, "OR021": 6,
    "B007": 7,  "B014": 8,  "B021": 9,
}

NUM_CLASSES = 10


def get_label(fault_type: str, fault_size: str = None) -> int:
    """Map fault type + size to integer class label (0-9)."""
    if fault_type == "Normal":
        return _LABEL_MAP["Normal"]
    return _LABEL_MAP[f"{fault_type}{fault_size}"]


def download_cwru(save_dir: str = "data/raw/CWRU") -> None:
    """
    Download CWRU dataset by cloning the GitHub mirror.

    Args:
        save_dir: Directory to save the dataset
    """
    save_path = Path(save_dir)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove if empty (from previous failed clone)
    if save_path.exists() and not any(save_path.iterdir()):
        import shutil
        shutil.rmtree(save_path)

    if save_path.exists() and (save_path / "op_0").exists():
        print(f"Dataset already exists at {save_dir}")
        return

    print(f"Cloning CWRU dataset from GitHub mirror...")
    print(f"  Repo: {_MIRROR_REPO}")
    print(f"  To:   {save_dir}")
    print(f"  This may take a few minutes (~200MB)...")

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", _MIRROR_REPO, str(save_path)],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        print("ERROR: git not found. Install Git or download manually:")
        print(f"  {_MIRROR_REPO}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Clone failed: {e.stderr}")
        print(f"\nManual download: visit {_MIRROR_REPO} → Code → Download ZIP")
        print(f"Then extract to: {save_dir}")
        sys.exit(1)

    # Verify
    for ld in range(4):
        op_dir = save_path / f"op_{ld}"
        if not op_dir.exists():
            print(f"WARNING: {op_dir} not found — dataset may be incomplete")

    print(f"Done. Dataset saved to {save_dir}")
    _print_summary(save_path)


def _print_summary(save_path: Path):
    """Print a summary of the downloaded dataset."""
    import os
    total_mat = 0
    class_count = {}
    for root, _, files in os.walk(save_path):
        for f in files:
            if f.endswith(".mat"):
                total_mat += 1
                rel = Path(root).relative_to(save_path)
                class_count[str(rel)] = class_count.get(str(rel), 0) + 1

    print(f"\n{'='*50}")
    print(f"Total .mat files: {total_mat}")
    print(f"Classes found:")
    for cls, cnt in sorted(class_count.items()):
        print(f"  {cls}: {cnt} files")
    print(f"{'='*50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download CWRU bearing dataset")
    parser.add_argument("--save_dir", type=str, default="data/raw/CWRU")
    args = parser.parse_args()

    print("=" * 60)
    print("CWRU Bearing Dataset Downloader")
    print("=" * 60)
    download_cwru(args.save_dir)
