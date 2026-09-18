"""
download_hockey_dataset.py - Download and Organize the Hockey Fight Benchmark Dataset
Downloads the official Hockey Fight dataset (1,000 clips: 500 Fight, 500 Non-Fight)
from HuggingFace mirror and structures it into train and validation sets.
"""

import os
import sys
import zipfile
import shutil
import random
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download


REPO_ID = "34data/hockey-fight-videos"
ARCHIVES = ["data_001.zip", "data_002.zip"]


def download_and_extract_hockey_dataset(target_dir="dataset/hockey", max_clips=None, val_ratio=0.2):
    """
    Downloads Hockey Fight videos from HuggingFace, extracts them, and organizes into:
      target_dir/
        train/
          violence/
          non_violence/
        val/
          violence/
          non_violence/
    """
    target_path = Path(target_dir)
    train_v = target_path / "train" / "violence"
    train_nv = target_path / "train" / "non_violence"
    val_v = target_path / "val" / "violence"
    val_nv = target_path / "val" / "non_violence"

    for d in [train_v, train_nv, val_v, val_nv]:
        d.mkdir(parents=True, exist_ok=True)

    # Check if dataset is already populated
    existing_videos = list(target_path.glob("*/*/*.avi")) + list(target_path.glob("*/*/*.mp4"))
    if len(existing_videos) >= 50 and (max_clips is None or len(existing_videos) >= max_clips):
        print(f"[Dataset] Hockey Fight dataset already exists at '{target_path}' ({len(existing_videos)} videos found).")
        return target_dir

    print(f"[Download] Fetching Hockey Fight dataset from Hugging Face ({REPO_ID})...")
    raw_extract_dir = target_path / "_raw_temp"
    raw_extract_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract each archive
    extracted_files = []
    for archive_name in ARCHIVES:
        print(f"[Download] Fetching {archive_name}...")
        try:
            zip_path = hf_hub_download(repo_id=REPO_ID, filename=archive_name, repo_type="dataset")
            print(f"[Extract] Extracting {archive_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith(('.avi', '.mp4')):
                        zip_ref.extract(file_info, raw_extract_dir)
                        extracted_files.append(raw_extract_dir / file_info.filename)
        except Exception as e:
            print(f"[Warning] Error downloading {archive_name}: {e}")

    # Separate into fight and no-fight
    # The dataset uses naming convention: fi*.avi (Fight/Violence) and no*.avi (No fight/Non-violence)
    all_raw_videos = list(raw_extract_dir.glob("*.avi")) + list(raw_extract_dir.glob("*.mp4"))
    fights = [f for f in all_raw_videos if f.name.lower().startswith("fi")]
    non_fights = [f for f in all_raw_videos if f.name.lower().startswith("no")]

    print(f"[Found] {len(fights)} fight clips and {len(non_fights)} non-fight clips.")

    # Limit clips if requested
    random.seed(42)
    random.shuffle(fights)
    random.shuffle(non_fights)

    if max_clips is not None:
        half = max_clips // 2
        fights = fights[:half]
        non_fights = non_fights[:half]
        print(f"[Subsetting] Limited dataset to {len(fights)} fights and {len(non_fights)} non-fights.")

    # Split into train and val
    def split_and_copy(file_list, train_dir, val_dir):
        split_idx = int(len(file_list) * (1.0 - val_ratio))
        train_files = file_list[:split_idx]
        val_files = file_list[split_idx:]

        for f in train_files:
            shutil.copy2(f, train_dir / f.name)
        for f in val_files:
            shutil.copy2(f, val_dir / f.name)

        return len(train_files), len(val_files)

    n_train_v, n_val_v = split_and_copy(fights, train_v, val_v)
    n_train_nv, n_val_nv = split_and_copy(non_fights, train_nv, val_nv)

    # Clean up temp folder
    shutil.rmtree(raw_extract_dir, ignore_errors=True)

    print("=" * 60)
    print("      HOCKEY FIGHT DATASET READY FOR TRAINING")
    print("=" * 60)
    print(f"Location: {target_path}")
    print(f"  - Train Set: {n_train_v} violence, {n_train_nv} non-violence (Total: {n_train_v + n_train_nv})")
    print(f"  - Val Set:   {n_val_v} violence, {n_val_nv} non-violence (Total: {n_val_v + n_val_nv})")
    print("=" * 60)

    return str(target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare Hockey Fight dataset")
    parser.add_argument("--target_dir", type=str, default="dataset/hockey", help="Destination directory")
    parser.add_argument("--max_clips", type=int, default=None, help="Max total clips (e.g. 200 or None for all 1000)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set split ratio (default 0.2)")

    args = parser.parse_args()
    download_and_extract_hockey_dataset(
        target_dir=args.target_dir,
        max_clips=args.max_clips,
        val_ratio=args.val_ratio
    )
