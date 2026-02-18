#!/usr/bin/env python3
"""
Download ISIC 2020 Skin Lesion Dataset for medical model training.

Usage:
    python scripts/models/download_isic_data.py [--output-dir data/isic2020]

This script:
1. Downloads the ISIC 2020 training metadata CSV
2. Downloads a configurable subset of images (default: 5000 balanced)
3. Organizes into train/val split directories by class
4. Reports class distribution

Dataset: ISIC 2020 Challenge (https://challenge2020.isic-archive.com/)
License: CC BY-NC 4.0
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Suppress TF import (not needed here)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def check_dependencies():
    missing = []
    try:
        import requests
    except ImportError:
        missing.append("requests")
    try:
        import pandas as pd
    except ImportError:
        missing.append("pandas")
    try:
        from PIL import Image
    except ImportError:
        missing.append("pillow")
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def download_isic_metadata(output_dir: Path) -> Path:
    """Download ISIC 2020 training labels CSV."""
    import requests

    metadata_url = (
        "https://isic-challenge-data.s3.amazonaws.com/2020/"
        "ISIC_2020_Training_GroundTruth.csv"
    )
    csv_path = output_dir / "ISIC_2020_Training_GroundTruth.csv"

    if csv_path.exists():
        print(f"  Metadata CSV already exists: {csv_path}")
        return csv_path

    print(f"  Downloading metadata from ISIC S3...")
    response = requests.get(metadata_url, stream=True, timeout=60)
    response.raise_for_status()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"  Saved: {csv_path}")
    return csv_path


def download_images_via_api(
    image_ids: list,
    output_dir: Path,
    max_images: int = 5000,
) -> dict:
    """
    Download images from ISIC Archive API.

    Returns dict mapping image_id -> local path.
    """
    import requests

    ISIC_API = "https://api.isic-archive.com/api/v2/images"
    downloaded = {}
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    to_download = image_ids[:max_images]
    total = len(to_download)

    print(f"  Downloading {total} images from ISIC S3...")

    for i, image_id in enumerate(to_download):
        img_path = images_dir / f"{image_id}.jpg"
        if img_path.exists():
            downloaded[image_id] = img_path
            continue

        try:
            # Direct S3 thumbnail URL (256px) — the /thumbnail/ API endpoint was removed
            url = f"https://isic-archive.s3.amazonaws.com/thumbnails/{image_id}_thumbnail.jpg"
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(img_path, "wb") as f:
                    f.write(r.content)
                downloaded[image_id] = img_path
            else:
                print(f"  WARNING: Could not download {image_id} (status {r.status_code})")
        except Exception as e:
            print(f"  WARNING: Error downloading {image_id}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{total}")

    print(f"  Downloaded {len(downloaded)}/{total} images")
    return downloaded


def organize_dataset(
    metadata_csv: Path,
    downloaded: dict,
    output_dir: Path,
    val_split: float = 0.2,
    max_per_class: int = 2500,
):
    """Organize images into train/val/class directories."""
    import pandas as pd
    import shutil
    import random

    df = pd.read_csv(metadata_csv)

    # ISIC 2020 columns: image_name, patient_id, sex, age_approx, etc., target (0=benign, 1=melanoma)
    df = df[df["image_name"].isin(downloaded.keys())].copy()

    class_map = {0: "benign", 1: "melanoma"}

    splits = {"train": {0: [], 1: []}, "val": {0: [], 1: []}}

    for target, group in df.groupby("target"):
        ids = group["image_name"].tolist()
        random.shuffle(ids)
        ids = ids[:max_per_class]
        n_val = max(1, int(len(ids) * val_split))
        splits["val"][target] = ids[:n_val]
        splits["train"][target] = ids[n_val:]

    total_copied = 0
    for split, class_data in splits.items():
        for target, ids in class_data.items():
            class_name = class_map[target]
            dest_dir = output_dir / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_id in ids:
                src = downloaded[img_id]
                dst = dest_dir / f"{img_id}.jpg"
                if not dst.exists():
                    shutil.copy2(src, dst)
                total_copied += 1

    print(f"  Organized {total_copied} images into train/val split")

    # Print summary
    summary = {}
    for split in ["train", "val"]:
        summary[split] = {}
        for cls in ["benign", "melanoma"]:
            n = len(list((output_dir / split / cls).glob("*.jpg")))
            summary[split][cls] = n
    return summary


def main():
    parser = argparse.ArgumentParser(description="Download ISIC 2020 dataset")
    parser.add_argument(
        "--output-dir",
        default="data/isic2020",
        help="Output directory for dataset (default: data/isic2020)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5000,
        help="Max images to download per class (default: 5000)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ISIC 2020 Dataset Downloader")
    print("=" * 70)
    print()

    check_dependencies()
    import pandas as pd

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Downloading metadata...")
    metadata_csv = download_isic_metadata(output_dir)

    print("\nStep 2: Reading metadata...")
    df = pd.read_csv(metadata_csv)
    print(f"  Total images in dataset: {len(df)}")
    print(f"  Melanoma: {df['target'].sum()}")
    print(f"  Benign:   {(df['target'] == 0).sum()}")

    # Balance classes: up to max_images melanoma + equal benign
    melanoma_ids = df[df["target"] == 1]["image_name"].tolist()
    benign_ids = df[df["target"] == 0]["image_name"].tolist()

    import random
    random.shuffle(melanoma_ids)
    random.shuffle(benign_ids)

    # Use all melanoma (limited) + same number of benign
    n_mel = min(len(melanoma_ids), args.max_images)
    selected_ids = melanoma_ids[:n_mel] + benign_ids[:n_mel]
    random.shuffle(selected_ids)

    print(f"\nStep 3: Downloading {len(selected_ids)} images (balanced)...")
    downloaded = download_images_via_api(selected_ids, output_dir, max_images=len(selected_ids))

    print("\nStep 4: Organizing into train/val split...")
    summary = organize_dataset(
        metadata_csv,
        downloaded,
        output_dir,
        val_split=args.val_split,
        max_per_class=n_mel,
    )

    print("\nDataset Summary:")
    print(json.dumps(summary, indent=2))

    # Save summary
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset saved to: {output_dir.resolve()}")
    print(f"Use this path with train_medical.py: --data-dir {output_dir}")
    print("\n" + "=" * 70)
    print("SUCCESS: ISIC 2020 dataset downloaded and organized!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
