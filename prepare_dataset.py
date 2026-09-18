"""
prepare_dataset.py - Dataset Preparation & Sample Generator for Violence Detection
Allows immediate testing with synthetic/demo videos and structuring custom or open-source datasets.
"""

import os
import sys
import argparse
import random
import cv2
import numpy as np
from pathlib import Path


def generate_synthetic_video(output_path, is_violent=False, num_frames=32, width=224, height=224, fps=15):
    """
    Generates a synthetic MP4 video clip:
    - Non-violent: Smooth, calm, predictable moving shapes (representing normal behavior).
    - Violent: Rapid, erratic, high-speed collisions, jittering shapes and flashes (representing fighting/violent action).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Initialize positions of simulated agents/objects
    pos1 = [width // 4, height // 2]
    pos2 = [3 * width // 4, height // 2]
    vel1 = [random.uniform(2, 4), random.uniform(-2, 2)]
    vel2 = [random.uniform(-4, -2), random.uniform(-2, 2)]

    for t in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40  # dark gray background

        if is_violent:
            # Erratic rapid motion, sudden direction changes, clashes
            vel1[0] += random.uniform(-6, 6)
            vel1[1] += random.uniform(-6, 6)
            vel2[0] += random.uniform(-6, 6)
            vel2[1] += random.uniform(-6, 6)

            # Cap max velocity
            vel1[0] = np.clip(vel1[0], -15, 15)
            vel1[1] = np.clip(vel1[1], -15, 15)
            vel2[0] = np.clip(vel2[0], -15, 15)
            vel2[1] = np.clip(vel2[1], -15, 15)

            # Draw erratic interaction lines/impact sparks
            if random.random() < 0.6:
                cv2.line(
                    frame,
                    (int(pos1[0]), int(pos1[1])),
                    (int(pos2[0]), int(pos2[1])),
                    (0, random.randint(150, 255), 255),
                    random.randint(2, 5)
                )
            color1 = (0, 0, 255)  # Red aggressive
            color2 = (0, 165, 255)  # Orange
        else:
            # Smooth gentle periodic movement
            vel1 = [3 * np.cos(t * 0.2), 2 * np.sin(t * 0.2)]
            vel2 = [-3 * np.cos(t * 0.2), -2 * np.sin(t * 0.2)]
            color1 = (0, 200, 0)  # Green calm
            color2 = (200, 200, 0)  # Cyan

        pos1[0] = (pos1[0] + vel1[0]) % width
        pos1[1] = (pos1[1] + vel1[1]) % height
        pos2[0] = (pos2[0] + vel2[0]) % width
        pos2[1] = (pos2[1] + vel2[1]) % height

        # Draw simulated subjects
        cv2.circle(frame, (int(pos1[0]), int(pos1[1])), 18, color1, -1)
        cv2.circle(frame, (int(pos2[0]), int(pos2[1])), 18, color2, -1)

        out.write(frame)

    out.release()


def create_sample_dataset(dataset_dir="dataset", train_count=10, val_count=4):
    """
    Creates a sample balanced dataset of synthetic videos for immediate training testing.
    """
    dataset_path = Path(dataset_dir)
    print(f"Creating sample dataset in '{dataset_path}'...")

    for split, count in [("train", train_count), ("val", val_count)]:
        v_dir = dataset_path / split / "violence"
        nv_dir = dataset_path / split / "non_violence"
        v_dir.mkdir(parents=True, exist_ok=True)
        nv_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            generate_synthetic_video(v_dir / f"violence_{i:03d}.mp4", is_violent=True)
            generate_synthetic_video(nv_dir / f"non_violence_{i:03d}.mp4", is_violent=False)

    print(f"[Done] Sample dataset ready:")
    print(f"  - Train: {train_count} violence, {train_count} non-violence videos")
    print(f"  - Val:   {val_count} violence, {val_count} non-violence videos")


def print_public_dataset_guide():
    """Prints instructions for popular benchmark violence datasets."""
    guide = """
========================================================================
             RECOMMENDED VIOLENCE DETECTION DATASETS
========================================================================

1. Real Life Violence Situations Dataset (Kaggle):
   - 2,000 real-world CCTV and mobile videos (1,000 Violence, 1,000 Non-Violence).
   - Kaggle URL: https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
   - Download & extract to:
       dataset/
         ├── violence/
         └── non_violence/

2. RWF-2000 (Real-World Fight) Dataset:
   - 2,000 video clips captured from surveillance cameras.
   - GitHub / Kaggle: https://github.com/mlama/RWF-2000

3. Hockey Fight Dataset:
   - 1,000 short clips (500 fights, 500 normal play).
   - Fast to download and benchmark.

Usage with this pipeline:
  Simply place video clips in the 'dataset' folder:
    dataset/train/violence/*.mp4
    dataset/train/non_violence/*.mp4
    dataset/val/violence/*.mp4
    dataset/val/non_violence/*.mp4
  (Or simply 'dataset/violence' and 'dataset/non_violence' - the loader will auto-split!)
========================================================================
    """
    print(guide)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for violence detection")
    parser.add_argument("--create_sample", action="store_true", help="Generate synthetic sample dataset for verification")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Target dataset directory")
    parser.add_argument("--train_count", type=int, default=12, help="Number of training clips per class")
    parser.add_argument("--val_count", type=int, default=4, help="Number of validation clips per class")
    parser.add_argument("--guide", action="store_true", help="Show public dataset download recommendations")

    args = parser.parse_args()

    if args.guide or (len(sys.argv) == 1):
        print_public_dataset_guide()

    if args.create_sample:
        create_sample_dataset(
            dataset_dir=args.dataset_dir,
            train_count=args.train_count,
            val_count=args.val_count
        )
