"""
dataset.py - Video Violence Dataset Loader & Preprocessing
Supports .mp4, .avi, .mov, .mkv, .webm files with uniform temporal sampling and augmentation.
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# Standard ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

VALID_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}


def extract_frames_from_video(video_path, num_frames=16, img_size=112):
    """
    Uniformly extracts `num_frames` from a video file, resized to (img_size, img_size).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames <= 0:
        # Fallback: sequential read if frame count unavailable
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(frame, (img_size, img_size)))
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Video contains no readable frames: {video_path}")

        # Subsample or pad
        if len(frames) >= num_frames:
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames + [frames[-1]] * (num_frames - len(frames))
        return sampled_frames

    # Standard case: sample frame indices uniformly
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)

    target_idx_set = set(indices)
    frame_dict = {}
    current_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in target_idx_set:
            frame_resized = cv2.resize(frame, (img_size, img_size))
            frame_dict[current_idx] = frame_resized
            if len(frame_dict) == len(target_idx_set):
                break
        current_idx += 1

    cap.release()

    # Fill any missing indices with nearest available frame
    sampled_frames = []
    last_valid = None
    for idx in indices:
        if idx in frame_dict:
            last_valid = frame_dict[idx]
        elif last_valid is None and frame_dict:
            last_valid = next(iter(frame_dict.values()))
        elif last_valid is None:
            last_valid = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        sampled_frames.append(last_valid)

    return sampled_frames


class ViolenceDataset(Dataset):
    """
    PyTorch Dataset for Video Violence Detection.
    """
    def __init__(self, samples, num_frames=16, img_size=112, is_training=False):
        """
        Args:
            samples: List of tuples [(video_path, label_int), ...]
                     label_int: 1 for violence, 0 for non-violence
            num_frames: Number of frames per video clip
            img_size: Spatial resolution (H = W)
            is_training: Apply data augmentation if True
        """
        self.samples = samples
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_training = is_training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        try:
            frames = extract_frames_from_video(
                video_path,
                num_frames=self.num_frames,
                img_size=self.img_size
            )
        except Exception as e:
            # Fallback for corrupted video: black frames
            print(f"[Warning] Failed loading {video_path}: {e}. Using zero frames.")
            frames = [np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8) for _ in range(self.num_frames)]

        # Video-level data augmentation (consistent across all frames in clip)
        do_flip = self.is_training and (random.random() < 0.5)
        brightness_factor = 1.0
        if self.is_training and (random.random() < 0.5):
            brightness_factor = random.uniform(0.8, 1.2)

        processed_frames = []
        for frame in frames:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if do_flip:
                frame = cv2.flip(frame, 1)

            frame = frame.astype(np.float32) / 255.0
            if brightness_factor != 1.0:
                frame = np.clip(frame * brightness_factor, 0.0, 1.0)

            # Normalize with ImageNet mean & std
            frame = (frame - IMAGENET_MEAN) / IMAGENET_STD

            # Convert from (H, W, C) to (C, H, W)
            frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float()
            processed_frames.append(frame_tensor)

        # Stack into (T, C, H, W)
        video_tensor = torch.stack(processed_frames, dim=0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return video_tensor, label_tensor


def scan_directory_for_videos(directory):
    """Finds all video files inside a directory."""
    directory = Path(directory)
    videos = []
    for ext in VALID_VIDEO_EXTENSIONS:
        videos.extend(list(directory.glob(f"*{ext}")))
        videos.extend(list(directory.glob(f"*{ext.upper()}")))
    return sorted(videos)


def collect_dataset_samples(dataset_dir):
    """
    Scans dataset_dir supporting:
    Structure A (pre-split):
      dataset_dir/
        ├── train/
        │   ├── violence/
        │   └── non_violence/
        └── val/
            ├── violence/
            └── non_violence/
    Structure B (flat classes):
      dataset_dir/
        ├── violence/
        └── non_violence/
    """
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    def match_class_folders(base_path):
        violence_names = ['violence', 'violent', 'fight', 'fights']
        non_violence_names = ['non_violence', 'non-violence', 'nonviolent', 'non_violent', 'normal', 'no_fight', 'nofight']

        v_folder = None
        nv_folder = None

        for sub in base_path.iterdir():
            if sub.is_dir():
                lower_name = sub.name.lower()
                if any(k in lower_name for k in non_violence_names):
                    nv_folder = sub
                elif any(k in lower_name for k in violence_names):
                    v_folder = sub

        return v_folder, nv_folder

    # Check for Structure A
    if train_dir.exists() and val_dir.exists():
        v_train, nv_train = match_class_folders(train_dir)
        v_val, nv_val = match_class_folders(val_dir)

        train_samples = []
        val_samples = []

        if v_train:
            train_samples.extend([(p, 1) for p in scan_directory_for_videos(v_train)])
        if nv_train:
            train_samples.extend([(p, 0) for p in scan_directory_for_videos(nv_train)])

        if v_val:
            val_samples.extend([(p, 1) for p in scan_directory_for_videos(v_val)])
        if nv_val:
            val_samples.extend([(p, 0) for p in scan_directory_for_videos(nv_val)])

        return train_samples, val_samples

    # Check for Structure B
    v_folder, nv_folder = match_class_folders(dataset_dir)
    all_v = [(p, 1) for p in scan_directory_for_videos(v_folder)] if v_folder else []
    all_nv = [(p, 0) for p in scan_directory_for_videos(nv_folder)] if nv_folder else []

    random.seed(42)
    random.shuffle(all_v)
    random.shuffle(all_nv)

    split_v = int(len(all_v) * 0.8)
    split_nv = int(len(all_nv) * 0.8)

    train_samples = all_v[:split_v] + all_nv[:split_nv]
    val_samples = all_v[split_v:] + all_nv[split_nv:]

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


def create_dataloaders(
    dataset_dir,
    batch_size=4,
    num_frames=16,
    img_size=112,
    num_workers=0
):
    """
    Creates PyTorch DataLoaders for training and validation.
    """
    train_samples, val_samples = collect_dataset_samples(dataset_dir)

    print(f"[Dataset] Found {len(train_samples)} training videos and {len(val_samples)} validation videos.")
    if len(train_samples) == 0:
        raise ValueError(
            f"No video samples found in '{dataset_dir}'! "
            "Please ensure videos are placed in 'violence' and 'non_violence' folders."
        )

    train_dataset = ViolenceDataset(
        train_samples,
        num_frames=num_frames,
        img_size=img_size,
        is_training=True
    )
    val_dataset = ViolenceDataset(
        val_samples if len(val_samples) > 0 else train_samples,
        num_frames=num_frames,
        img_size=img_size,
        is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=len(train_dataset) > batch_size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader
