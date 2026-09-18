"""
violence_detector.py - Real-Time and Video File Violence Detection System
Supports both:
  1. Live Webcam Detection
  2. Recorded Video File Detection (with sample clip selection, progress tracking, and summary report)
Powered by trained PyTorch Deep Learning Model (MobileNetV2 + BiLSTM) with optical flow fallback.
"""

import os
import sys
import time
import argparse
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch

from model import ViolenceDetectionModel

# ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class HeuristicViolenceDetector:
    """Fallback detector using Optical Flow and Face Detection."""
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.prev_gray = None

    def predict(self, frame):
        features = {'rapid_movement': False, 'multiple_faces': False}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 1:
            features['multiple_faces'] = True

        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_magnitude = np.mean(magnitude)
            if avg_magnitude > 8.0:
                features['rapid_movement'] = True

        self.prev_gray = gray
        score = (0.6 if features['rapid_movement'] else 0.0) + (0.4 if features['multiple_faces'] else 0.0)
        return score, (score > 0.5)


class DeepViolenceDetector:
    """Deep Learning Violence Detector using trained PyTorch model."""
    def __init__(self, model_path="best_violence_model.pt", device=None, num_frames=16, img_size=112):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_frames = num_frames
        self.img_size = img_size
        self.frame_buffer = deque(maxlen=num_frames)
        self.model = None
        self.is_loaded = False
        self.smoothed_score = 0.0
        self.alpha = 0.3  # Exponential moving average smoothing factor

        if os.path.exists(model_path):
            try:
                print(f"[Detector] Loading trained model weights from: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)

                self.model = ViolenceDetectionModel(use_torchvision=True).to(self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.num_frames = checkpoint.get('num_frames', self.num_frames)
                    self.img_size = checkpoint.get('img_size', self.img_size)
                    self.frame_buffer = deque(maxlen=self.num_frames)
                else:
                    self.model.load_state_dict(checkpoint)

                self.model.eval()
                self.is_loaded = True
                print(f"[Detector] Model successfully loaded on {self.device}!")
            except Exception as e:
                print(f"[Warning] Failed loading model checkpoint: {e}")
                self.is_loaded = False
        else:
            print(f"[Info] Model checkpoint '{model_path}' not found. Run 'python train.py' to train weights.")

        self.heuristic = HeuristicViolenceDetector()

    def preprocess_frame(self, frame):
        """Resizes and normalizes frame to PyTorch tensor (C, H, W)."""
        resized = cv2.resize(frame, (self.img_size, self.img_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(normalized.transpose(2, 0, 1)).float()

    def process_frame(self, frame):
        """
        Receives current frame, updates buffer, runs inference, and returns (score, is_violent, mode_name).
        """
        if self.is_loaded:
            preprocessed = self.preprocess_frame(frame)
            self.frame_buffer.append(preprocessed)

            if len(self.frame_buffer) == self.num_frames:
                clip_tensor = torch.stack(list(self.frame_buffer), dim=0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    prob = self.model.predict_proba(clip_tensor).item()

                self.smoothed_score = self.alpha * prob + (1.0 - self.alpha) * self.smoothed_score
                is_violent = self.smoothed_score >= 0.5
                return self.smoothed_score, is_violent, "Deep Learning (Hockey Fight Model)"
            else:
                return self.smoothed_score, (self.smoothed_score >= 0.5), f"Buffering ({len(self.frame_buffer)}/{self.num_frames})"
        else:
            raw_score, is_violent = self.heuristic.predict(frame)
            self.smoothed_score = self.alpha * raw_score + (1.0 - self.alpha) * self.smoothed_score
            return self.smoothed_score, (self.smoothed_score >= 0.5), "Heuristic Fallback (Optical Flow)"


def draw_hud(frame, score, is_violent, mode_str, fps, progress_str=None):
    """Draws an informative, modern Heads-Up Display (HUD) on the video frame."""
    h, w = frame.shape[:2]

    # Color palette
    if is_violent:
        status_text = "ALERT: VIOLENCE DETECTED!"
        theme_color = (0, 0, 255)  # Red
    else:
        status_text = "STATUS: NORMAL / SAFE"
        theme_color = (0, 200, 0)  # Green

    # Outer alert border
    border_thick = 8 if is_violent else 3
    cv2.rectangle(frame, (0, 0), (w, h), theme_color, border_thick)

    # Top overlay header banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Status title
    cv2.putText(frame, status_text, (15, 34), cv2.FONT_HERSHEY_DUPLEX, 0.9, theme_color, 2)

    # Subtext: Mode, FPS, Progress
    subtext = f"Mode: {mode_str} | FPS: {fps:.1f}"
    if progress_str:
        subtext += f" | {progress_str}"
    cv2.putText(frame, subtext, (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)

    # Confidence Gauge Bar (right side)
    gauge_w, gauge_h = 200, 22
    gauge_x, gauge_y = w - gauge_w - 20, 18
    cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), (50, 50, 50), -1)
    fill_w = int(np.clip(score, 0.0, 1.0) * gauge_w)
    cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + fill_w, gauge_y + gauge_h), theme_color, -1)
    cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), (220, 220, 220), 1)

    score_label = f"Confidence: {score * 100:.1f}%"
    cv2.putText(frame, score_label, (gauge_x + 35, gauge_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Bottom navigation hints
    cv2.putText(frame, "[Q] Quit  |  [P] Pause/Resume  |  [S] Save Screenshot  |  [R] Replay",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    return frame


def run_detection(source, model_path="best_violence_model.pt", device=None, save_output=None, headless=False):
    """Executes the violence detection loop on a camera feed or video file."""
    is_camera = isinstance(source, int) or str(source).isdigit()
    source_val = int(source) if is_camera else str(source)

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"[Error] Could not open video source: {source}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else 0
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if video_fps <= 0 or np.isnan(video_fps):
        video_fps = 25

    detector = DeepViolenceDetector(model_path=model_path, device=device)

    # Video writer for saving output
    writer = None
    if save_output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_output, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (w, h))
        print(f"[Writer] Saving output video to: {save_output}")

    title_str = "Live Webcam Violence Monitor" if is_camera else f"Video Analysis - {Path(str(source)).name}"
    print("\n" + "=" * 60)
    print(f"       {title_str.upper()}")
    print("=" * 60)
    print("Controls: 'Q' Quit | 'P' Pause | 'S' Screenshot | 'R' Replay")

    frame_idx = 0
    violent_frames = 0
    peak_score = 0.0
    violent_intervals = []
    in_violence_event = False
    event_start = 0.0

    screenshot_count = 0
    prev_time = time.time()
    fps = 0.0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if not is_camera:
                    print("\n[Complete] Reached end of video file.")
                    break
                else:
                    time.sleep(0.01)
                    continue

            frame_idx += 1
            if is_camera:
                frame = cv2.flip(frame, 1)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            score, is_violent, mode_str = detector.process_frame(frame)
            peak_score = max(peak_score, score)

            # Track violence timestamps
            current_sec = frame_idx / video_fps
            if is_violent:
                violent_frames += 1
                if not in_violence_event:
                    in_violence_event = True
                    event_start = current_sec
            else:
                if in_violence_event:
                    in_violence_event = False
                    violent_intervals.append((event_start, current_sec))

            # Progress info for video file
            progress_str = None
            if total_frames > 0:
                pct = int((frame_idx / total_frames) * 100)
                sec_curr = int(current_sec)
                sec_tot = int(total_frames / video_fps)
                progress_str = f"Progress: {frame_idx}/{total_frames} ({pct}%) [{sec_curr//60:02d}:{sec_curr%60:02d}/{sec_tot//60:02d}:{sec_tot%60:02d}]"

            annotated_frame = draw_hud(frame, score, is_violent, mode_str, fps, progress_str)

            if writer:
                writer.write(annotated_frame)

            if not headless:
                cv2.imshow("AI Violence Detection", annotated_frame)
                key = cv2.waitKey(1 if is_camera else int(1000 / video_fps)) & 0xFF
            else:
                key = 0xFF

        if not headless:
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f"screenshot_{int(time.time())}_{screenshot_count}.jpg"
                cv2.imwrite(screenshot_name, annotated_frame)
                print(f"[Screenshot] Saved image: {screenshot_name}")
                screenshot_count += 1
            elif key == ord('p'):
                paused = not paused
                print("[Info] Paused" if paused else "[Info] Resumed")
            elif key == ord('r') and not is_camera:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                detector.frame_buffer.clear()
                print("[Info] Replaying video from beginning...")

    if in_violence_event:
        violent_intervals.append((event_start, frame_idx / video_fps))

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Print summary report
    if not is_camera and frame_idx > 0:
        total_sec = frame_idx / video_fps
        violence_pct = (violent_frames / frame_idx) * 100
        print("\n" + "=" * 60)
        print("              DETECTION SUMMARY REPORT")
        print("=" * 60)
        print(f"Video File:         {source}")
        print(f"Total Duration:     {total_sec:.1f} seconds ({frame_idx} frames)")
        print(f"Peak Confidence:    {peak_score * 100:.1f}%")
        print(f"Violence Detected:  {'YES' if violent_frames > 0 else 'NO'}")
        print(f"Violent Frames:     {violent_frames}/{frame_idx} ({violence_pct:.1f}%)")
        if violent_intervals:
            print("Incident Intervals:")
            for start, end in violent_intervals:
                print(f"  - {start:.1f}s to {end:.1f}s (Duration: {end - start:.1f}s)")
        if save_output:
            print(f"Saved Output Video: {save_output}")
        print("=" * 60)


def interactive_menu():
    """Shows interactive selection menu when launched without command-line flags."""
    print("\n" + "=" * 60)
    print("        AI VIDEO VIOLENCE DETECTION SYSTEM")
    print("=" * 60)
    print("Please select a detection mode:")
    print("  [1] Live Webcam Detection")
    print("  [2] Recorded Video File Detection")
    print("=" * 60)

    choice = input("Enter choice (1 or 2, default 1): ").strip()
    if choice == "2":
        # Video File Option
        print("\nAvailable Sample Videos:")
        val_videos = []
        for p in [Path("dataset/hockey/val/violence"), Path("dataset/hockey/val/non_violence")]:
            if p.exists():
                val_videos.extend(list(p.glob("*.avi")) + list(p.glob("*.mp4")))

        if val_videos:
            for idx, vid in enumerate(val_videos[:8], 1):
                label = "Fight" if "violence" in str(vid.parent) else "Non-Fight"
                print(f"  [{idx}] {vid.name} ({label})")
            print(f"  [0] Enter custom video file path")

            sub_choice = input(f"Select a sample video (1-{min(8, len(val_videos))}) or 0 for custom path: ").strip()
            if sub_choice.isdigit() and 1 <= int(sub_choice) <= min(8, len(val_videos)):
                video_path = str(val_videos[int(sub_choice) - 1])
            else:
                video_path = input("Enter video file path: ").strip().strip('"')
        else:
            video_path = input("Enter video file path: ").strip().strip('"')

        save_choice = input("Save annotated output video? (y/N): ").strip().lower()
        save_out = "annotated_violence_detection.mp4" if save_choice == 'y' else None
        run_detection(video_path, save_output=save_out)
    else:
        # Webcam Option
        cam_idx = input("Enter camera index (default 0): ").strip()
        cam_source = int(cam_idx) if cam_idx.isdigit() else 0
        run_detection(cam_source)


def main():
    parser = argparse.ArgumentParser(description="Real-Time and Video File Violence Detection")
    parser.add_argument("--source", type=str, default=None, help="Camera index (0) or path to video file")
    parser.add_argument("--model", type=str, default="best_violence_model.pt", help="Path to trained model weights")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--save_output", type=str, default=None, help="Path to save annotated output video (.mp4)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")

    args = parser.parse_args()

    if args.source is None:
        interactive_menu()
    else:
        run_detection(
            source=args.source,
            model_path=args.model,
            device=args.device,
            save_output=args.save_output,
            headless=args.headless
        )


if __name__ == "__main__":
    main()