# fix_csv_paths.py
#
# Rewrites the image paths in driving_log.csv for each training data folder
# so they use paths relative to the project root instead of hardcoded absolute paths.
#
# Before: C:/Users/whoever/.../IMG/center_2026_04_09_12_15_18_849.jpg,0
# After:  training_data_backwards/IMG/center_2026_04_09_12_15_18_849.jpg,0

import csv
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

DATASETS = [
    # "training_data_forwards",
    # "training_data_backwards",
    "training_data_forwards_backwards_unstable",
]

for dataset in DATASETS:
    csv_path = SCRIPT_DIR / dataset / "driving_log.csv"

    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found")
        continue

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                filename = Path(row[0]).name          # e.g. center_2026_04_09_....jpg
                new_path = f"{dataset}/IMG/{filename}"
                rows.append([new_path] + row[1:])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[DONE] {csv_path} — {len(rows)} rows updated")