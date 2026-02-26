#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download stock market dataset from Kaggle and copy CSV files to the data directory.

Usage:
    python scripts/download_data.py
"""
import shutil
from pathlib import Path

import kagglehub

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"


def main():
    print("Downloading stock market dataset from Kaggle...")
    path = kagglehub.dataset_download("paultimothymooney/stock-market-data")
    print(f"Path to dataset files: {path}")

    dataset_path = Path(path)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old CSV files in the data directory
    for old_csv in DATA_DIR.glob("*.csv"):
        old_csv.unlink()
        print(f"Removed old file: {old_csv.name}")

    # Copy all CSV files from the downloaded dataset into data/
    csv_files = list(dataset_path.rglob("*.csv"))
    if not csv_files:
        print("Warning: No CSV files found in the downloaded dataset.")
        return

    for csv_file in csv_files:
        dest = DATA_DIR / csv_file.name
        if dest.exists():
            counter = 1
            stem = csv_file.stem
            suffix = csv_file.suffix
            while dest.exists():
                dest = DATA_DIR / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.copy2(csv_file, dest)
        print(f"Copied: {csv_file} -> {dest.name}")

    print(f"Copied {len(csv_files)} CSV file(s) to {DATA_DIR}")


if __name__ == "__main__":
    main()
