"""
train.py - Training Pipeline for Video Violence Detection Model
Trains spatial-temporal neural network on Hockey Fight and custom video datasets.
Includes feature caching for rapid training, evaluation metrics, and checkpointing.
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt

from model import ViolenceDetectionModel
from dataset import create_dataloaders


def extract_dataset_features(model, dataloader, device):
    """
    Extracts spatial CNN backbone features for all video clips in dataloader.
    Returns:
        all_features: Tensor of shape (N, T, feature_dim)
        all_labels: Tensor of shape (N, 1)
    """
    model.eval()
    feature_list = []
    label_list = []

    total_batches = len(dataloader)
    print(f"[Feature Extraction] Pre-extracting CNN features for {len(dataloader.dataset)} clips...")

    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(dataloader):
            # videos shape: (B, T, C, H, W)
            B, T, C, H, W = videos.shape
            videos = videos.to(device).contiguous().view(B * T, C, H, W)

            # Pass through CNN backbone
            feats = model.extract_features(videos)  # (B * T, feature_dim)
            feats = feats.view(B, T, -1).cpu()     # (B, T, feature_dim)

            feature_list.append(feats)
            label_list.append(labels.unsqueeze(1))

            if (batch_idx + 1) % max(1, total_batches // 5) == 0 or (batch_idx + 1) == total_batches:
                print(f"  Processed {batch_idx + 1}/{total_batches} batches...")

    all_features = torch.cat(feature_list, dim=0)
    all_labels = torch.cat(label_list, dim=0)
    return all_features, all_labels


def train_cached_epoch(model, dataloader, criterion, optimizer, device):
    """Trains sequence BiLSTM + Classifier using precomputed features."""
    model.temporal_model.train()
    model.attention.train()
    model.classifier.train()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model.classify_features(features)
        loss = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().flatten()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy().flatten())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def evaluate_cached(model, dataloader, criterion, device):
    """Evaluates sequence BiLSTM + Classifier on validation features."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model.classify_features(features)
            loss = criterion(logits, labels)

            running_loss += loss.item() * features.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy().flatten())

    val_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', zero_division=0
    )
    return val_loss, acc, precision, recall, f1


def plot_training_history(history, save_path="training_history.png"):
    """Generates and saves training and validation curve plots."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r--s', label='Val Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, history['val_acc'], 'g-o', label='Val Accuracy')
    ax2.plot(epochs, history['val_f1'], 'm--^', label='Val F1 Score')
    ax2.set_title('Validation Accuracy & F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Training history saved to: {save_path}")


def train(args):
    print("=" * 65)
    print("        DEEP LEARNING VIOLENCE DETECTION TRAINING")
    print("=" * 65)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Device] Using compute device: {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    # Load raw video dataloaders
    print(f"[Dataset] Loading clips from: {args.dataset_dir}")
    train_loader, val_loader = create_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        img_size=args.img_size,
        num_workers=args.num_workers
    )

    # Initialize model
    print("[Model] Initializing Spatial-Temporal Violence Detection Network...")
    model = ViolenceDetectionModel().to(device)

    # Fast Feature Caching Mode
    print("\n[Optimization] Feature caching enabled for rapid convergence...")
    t0 = time.time()
    train_feats, train_labels = extract_dataset_features(model, train_loader, device)
    val_feats, val_labels = extract_dataset_features(model, val_loader, device)
    print(f"[Features] Extraction completed in {time.time() - t0:.1f}s.")

    # Create cached DataLoaders
    cached_train_loader = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=args.batch_size,
        shuffle=True
    )
    cached_val_loader = DataLoader(
        TensorDataset(val_feats, val_labels),
        batch_size=args.batch_size,
        shuffle=False
    )

    # Loss & Optimizer for sequence head
    criterion = nn.BCEWithLogitsLoss()
    head_params = list(model.temporal_model.parameters()) + list(model.attention.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    print("\n" + "-" * 75)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val F1':<8} | {'Status'}")
    print("-" * 75)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_cached_epoch(model, cached_train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_prec, v_rec, v_f1 = evaluate_cached(model, cached_val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        history['val_f1'].append(v_f1)

        # Checkpoint if validation metric improves
        status = ""
        if v_f1 > best_val_f1 or (v_f1 == best_val_f1 and v_loss < best_val_loss):
            best_val_f1 = v_f1
            best_val_loss = v_loss

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': v_loss,
                'val_acc': v_acc,
                'val_f1': v_f1,
                'num_frames': args.num_frames,
                'img_size': args.img_size,
            }
            torch.save(checkpoint, args.model_save_path)
            status = f"* Saved ({args.model_save_path})"

        print(f"{epoch:<8} | {t_loss:<12.4f} | {t_acc*100:<9.1f}% | {v_loss:<10.4f} | {v_acc*100:<8.1f}% | {v_f1:<8.4f} | {status}")

    total_time = time.time() - start_time
    print("-" * 75)
    print(f"[Done] Training complete in {total_time:.1f}s.")
    print(f"[Best Performance] Val Accuracy: {history['val_acc'][history['val_f1'].index(best_val_f1)]*100:.1f}%, Val F1: {best_val_f1:.4f}")
    print(f"[Model Checkpoint] Saved to: {args.model_save_path}")

    # Generate curves plot
    plot_training_history(history, save_path=args.plot_save_path)

    # Save metadata
    metadata = {
        "dataset": args.dataset_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_frames": args.num_frames,
        "img_size": args.img_size,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "final_val_acc": history['val_acc'][-1]
    }
    with open("training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("[Metadata] Saved training_metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Video Violence Detection Model")
    parser.add_argument("--dataset_dir", type=str, default="dataset/hockey", help="Path to video dataset directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames sampled per clip")
    parser.add_argument("--img_size", type=int, default=112, help="Input frame image size (H=W)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default 0 for Windows)")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--model_save_path", type=str, default="best_violence_model.pt", help="File to save best model")
    parser.add_argument("--plot_save_path", type=str, default="training_history.png", help="Path to save training plot")

    args = parser.parse_args()
    train(args)
