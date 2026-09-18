"""
gui_app.py - Modern Desktop GUI Front-End for AI Violence Detection
Double-clickable, clean graphical interface for Live Webcam & Video File Detection.
"""

import os
import sys
import time
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Make Windows DPI-aware for razor sharp text and rendering
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

from violence_detector import DeepViolenceDetector, draw_hud


# Color Scheme - Modern Dark Cybersecurity Aesthetic
BG_MAIN = "#0f172a"      # Slate 900
BG_PANEL = "#1e293b"     # Slate 800
BG_CARD = "#334155"      # Slate 700
TEXT_WHITE = "#f8fafc"
TEXT_MUTED = "#94a3b8"
ACCENT_CYAN = "#38bdf8"
ACCENT_GREEN = "#22c55e"
ACCENT_RED = "#ef4444"
BORDER_COLOR = "#475569"


class ViolenceDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Video Violence Detection System 🛡️")
        self.root.geometry("1240x820")
        self.root.minsize(1050, 700)
        self.root.configure(bg=BG_MAIN)

        # Application State
        self.detector = DeepViolenceDetector(model_path="best_violence_model.pt")
        self.cap = None
        self.is_running = False
        self.is_paused = False
        self.worker_thread = None
        self.current_source_name = "Idle"
        self.save_output = False
        self.video_writer = None
        self.screenshot_count = 0
        self.incident_count = 0
        self.last_incident_time = 0.0

        # Build GUI
        self.setup_styles()
        self.create_header()
        self.create_main_content()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Configure Scrollbar
        style.configure(
            "Dark.Vertical.TScrollbar",
            background=BG_CARD,
            troughcolor=BG_PANEL,
            bordercolor=BG_PANEL,
            arrowcolor=TEXT_MUTED
        )

    def create_header(self):
        header_frame = tk.Frame(self.root, bg=BG_PANEL, height=65)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        # Title & Subtitle
        title_box = tk.Frame(header_frame, bg=BG_PANEL)
        title_box.pack(side=tk.LEFT, padx=25, pady=10)

        title_lbl = tk.Label(
            title_box,
            text="AI VIDEO VIOLENCE DETECTION STATION",
            font=("Segoe UI", 16, "bold"),
            fg=TEXT_WHITE,
            bg=BG_PANEL
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            title_box,
            text="Deep Learning Spatial-Temporal Surveillance Monitor (MobileNetV2 + BiLSTM)",
            font=("Segoe UI", 9),
            fg=ACCENT_CYAN,
            bg=BG_PANEL
        )
        sub_lbl.pack(anchor="w")

        # Status Indicator in Header
        model_status_text = "● Model: Active (Hockey Fight 92.5%)" if self.detector.is_loaded else "● Model: Heuristic Fallback"
        model_status_color = ACCENT_GREEN if self.detector.is_loaded else "#f59e0b"

        self.header_status_lbl = tk.Label(
            header_frame,
            text=model_status_text,
            font=("Segoe UI", 10, "bold"),
            fg=model_status_color,
            bg=BG_PANEL
        )
        self.header_status_lbl.pack(side=tk.RIGHT, padx=25, pady=15)

    def create_main_content(self):
        main_container = tk.Frame(self.root, bg=BG_MAIN)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # ----------------- LEFT: Video Display & Controls -----------------
        left_frame = tk.Frame(main_container, bg=BG_MAIN)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        # Video Canvas Screen
        self.video_canvas = tk.Canvas(
            left_frame,
            bg="#020617",
            highlightthickness=2,
            highlightbackground=BORDER_COLOR
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.draw_placeholder()

        # Control Panel
        controls_panel = tk.Frame(left_frame, bg=BG_PANEL, height=80)
        controls_panel.pack(fill=tk.X, pady=(12, 0))

        # Button styling helper
        def make_btn(parent, text, cmd, bg_color, fg_color=TEXT_WHITE, width=13):
            btn = tk.Button(
                parent,
                text=text,
                command=cmd,
                font=("Segoe UI", 10, "bold"),
                bg=bg_color,
                fg=fg_color,
                activebackground="#475569",
                activeforeground=TEXT_WHITE,
                relief=tk.FLAT,
                bd=0,
                padx=12,
                pady=7,
                cursor="hand2",
                width=width
            )
            return btn

        # Left Buttons: Live Webcam & Video File
        self.btn_webcam = make_btn(controls_panel, "📹 Live Webcam", self.start_webcam, "#2563eb")
        self.btn_webcam.pack(side=tk.LEFT, padx=(15, 8), pady=15)

        self.btn_file = make_btn(controls_panel, "📁 Open Video", self.open_video_file, "#0d9488")
        self.btn_file.pack(side=tk.LEFT, padx=6, pady=15)

        self.btn_pause = make_btn(controls_panel, "⏸️ Pause", self.toggle_pause, "#64748b", width=9)
        self.btn_pause.pack(side=tk.LEFT, padx=6, pady=15)

        self.btn_stop = make_btn(controls_panel, "⏹️ Stop", self.stop_stream, "#475569", width=9)
        self.btn_stop.pack(side=tk.LEFT, padx=6, pady=15)

        # Right Buttons: Snapshot & Record toggle
        self.save_var = tk.BooleanVar(value=False)
        self.chk_save = tk.Checkbutton(
            controls_panel,
            text="💾 Save Annotated MP4",
            variable=self.save_var,
            font=("Segoe UI", 9),
            bg=BG_PANEL,
            fg=TEXT_WHITE,
            selectcolor=BG_CARD,
            activebackground=BG_PANEL,
            activeforeground=TEXT_WHITE,
            cursor="hand2"
        )
        self.chk_save.pack(side=tk.RIGHT, padx=12, pady=15)

        self.btn_snap = make_btn(controls_panel, "📸 Snapshot", self.take_snapshot, "#334155", width=10)
        self.btn_snap.pack(side=tk.RIGHT, padx=6, pady=15)

        # ----------------- RIGHT: Status & Incident Logs -----------------
        right_frame = tk.Frame(main_container, bg=BG_MAIN, width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_frame.pack_propagate(False)

        # 1. LIVE ALERT CARD
        self.alert_card = tk.Frame(right_frame, bg=BG_PANEL, highlightthickness=2, highlightbackground=ACCENT_GREEN)
        self.alert_card.pack(fill=tk.X, pady=(0, 12))

        alert_header = tk.Label(self.alert_card, text="CURRENT MONITOR STATUS", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=BG_PANEL)
        alert_header.pack(anchor="w", padx=15, pady=(12, 4))

        self.status_badge = tk.Label(
            self.alert_card,
            text="NORMAL / SAFE",
            font=("Segoe UI", 15, "bold"),
            fg=ACCENT_GREEN,
            bg=BG_PANEL
        )
        self.status_badge.pack(anchor="w", padx=15, pady=(0, 8))

        # Confidence Bar
        conf_lbl_frame = tk.Frame(self.alert_card, bg=BG_PANEL)
        conf_lbl_frame.pack(fill=tk.X, padx=15)
        
        tk.Label(conf_lbl_frame, text="Violence Confidence", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_PANEL).pack(side=tk.LEFT)
        self.conf_val_lbl = tk.Label(conf_lbl_frame, text="0.0%", font=("Segoe UI", 10, "bold"), fg=TEXT_WHITE, bg=BG_PANEL)
        self.conf_val_lbl.pack(side=tk.RIGHT)

        self.progress_canvas = tk.Canvas(self.alert_card, height=14, bg=BG_CARD, highlightthickness=0)
        self.progress_canvas.pack(fill=tk.X, padx=15, pady=(4, 12))

        # Stats Grid inside card
        stats_frame = tk.Frame(self.alert_card, bg=BG_PANEL)
        stats_frame.pack(fill=tk.X, padx=15, pady=(0, 12))

        self.lbl_fps = tk.Label(stats_frame, text="FPS: --", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_PANEL)
        self.lbl_fps.pack(side=tk.LEFT)

        self.lbl_frames = tk.Label(stats_frame, text="Frames: 0", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_PANEL)
        self.lbl_frames.pack(side=tk.RIGHT)

        # 2. QUICK TEST SAMPLES
        quick_frame = tk.LabelFrame(
            right_frame,
            text="  Quick Test Benchmark Videos  ",
            font=("Segoe UI", 9, "bold"),
            bg=BG_PANEL,
            fg=ACCENT_CYAN,
            bd=1,
            relief=tk.SOLID
        )
        quick_frame.pack(fill=tk.X, pady=(0, 12))

        btn_sample_fight = make_btn(
            quick_frame,
            "🥊 Test Hockey Fight (fi106)",
            lambda: self.load_sample_clip("dataset/hockey/val/violence/fi106_xvid.avi"),
            "#dc2626",
            width=28
        )
        btn_sample_fight.pack(padx=12, pady=(10, 6), fill=tk.X)

        btn_sample_normal = make_btn(
            quick_frame,
            "⛸️ Test Normal Play (no101)",
            lambda: self.load_sample_clip("dataset/hockey/val/non_violence/no101_xvid.avi"),
            "#16a34a",
            width=28
        )
        btn_sample_normal.pack(padx=12, pady=(0, 12), fill=tk.X)

        # 3. INCIDENT LOG PANEL
        log_frame = tk.LabelFrame(
            right_frame,
            text="  Incident Activity Log  ",
            font=("Segoe UI", 9, "bold"),
            bg=BG_PANEL,
            fg=TEXT_WHITE,
            bd=1,
            relief=tk.SOLID
        )
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_listbox = tk.Listbox(
            log_frame,
            bg=BG_MAIN,
            fg=TEXT_WHITE,
            font=("Consolas", 9),
            selectbackground=BG_CARD,
            highlightthickness=0,
            bd=0
        )
        self.log_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)

        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_listbox.yview, style="Dark.Vertical.TScrollbar")
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=8, padx=(0, 8))
        self.log_listbox.config(yscrollcommand=log_scroll.set)

        self.add_log_entry("System initialized. Monitoring engine ready.")

    def draw_placeholder(self):
        """Draws welcoming standby screen when video is idle."""
        self.video_canvas.delete("all")
        w = self.video_canvas.winfo_width() or 700
        h = self.video_canvas.winfo_height() or 480

        self.video_canvas.create_text(
            w // 2, h // 2 - 30,
            text="AI Surveillance Feed Standby",
            font=("Segoe UI", 18, "bold"),
            fill=TEXT_WHITE
        )
        self.video_canvas.create_text(
            w // 2, h // 2 + 15,
            text="Click 'Live Webcam' or 'Open Video' below to start detection",
            font=("Segoe UI", 11),
            fill=TEXT_MUTED
        )

    def add_log_entry(self, text, is_alert=False):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {text}"
        self.log_listbox.insert(tk.END, entry)
        if is_alert:
            self.log_listbox.itemconfig(tk.END, {'fg': ACCENT_RED})
        self.log_listbox.see(tk.END)

    def update_status_card(self, score, is_violent, fps, frame_count):
        """Updates the right-hand status badge and confidence meter."""
        pct = score * 100.0
        self.conf_val_lbl.config(text=f"{pct:.1f}%")
        self.lbl_fps.config(text=f"FPS: {fps:.1f}")
        self.lbl_frames.config(text=f"Frames: {frame_count}")

        # Update Progress Bar
        self.progress_canvas.delete("all")
        cw = self.progress_canvas.winfo_width() or 200
        bar_w = int(np.clip(score, 0.0, 1.0) * cw)
        bar_color = ACCENT_RED if is_violent else ACCENT_GREEN

        if bar_w > 0:
            self.progress_canvas.create_rectangle(0, 0, bar_w, 14, fill=bar_color, width=0)

        # Update Badge Color
        if is_violent:
            self.status_badge.config(text="🚨 VIOLENCE DETECTED!", fg=ACCENT_RED)
            self.alert_card.config(highlightbackground=ACCENT_RED)
        else:
            self.status_badge.config(text="🟢 NORMAL / SAFE", fg=ACCENT_GREEN)
            self.alert_card.config(highlightbackground=ACCENT_GREEN)

    def start_webcam(self):
        self.stop_stream()
        self.current_source_name = "Live Webcam (Device 0)"
        self.add_log_entry("Connecting to live webcam feed...")
        self.start_processing(0)

    def open_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All Files", "*.*")]
        )
        if file_path:
            self.stop_stream()
            self.current_source_name = Path(file_path).name
            self.add_log_entry(f"Loaded video: {self.current_source_name}")
            self.start_processing(file_path)

    def load_sample_clip(self, relative_path):
        full_path = Path(relative_path).resolve()
        if not full_path.exists():
            messagebox.showwarning("File Missing", f"Could not find sample video at:\n{full_path}")
            return
        self.stop_stream()
        self.current_source_name = full_path.name
        self.add_log_entry(f"Loading benchmark sample: {full_path.name}")
        self.start_processing(str(full_path))

    def toggle_pause(self):
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        self.btn_pause.config(text="▶️ Resume" if self.is_paused else "⏸️ Pause")
        self.add_log_entry("Paused" if self.is_paused else "Resumed")

    def take_snapshot(self):
        if hasattr(self, 'current_annotated_frame') and self.current_annotated_frame is not None:
            filename = f"snapshot_{int(time.time())}_{self.screenshot_count}.jpg"
            cv2.imwrite(filename, self.current_annotated_frame)
            self.screenshot_count += 1
            self.add_log_entry(f"Saved snapshot: {filename}")
            messagebox.showinfo("Snapshot Saved", f"Saved image:\n{filename}")
        else:
            messagebox.showinfo("Info", "No active frame to capture.")

    def start_processing(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Failed to open video stream:\n{source}")
            self.draw_placeholder()
            return

        self.is_running = True
        self.is_paused = False
        self.btn_pause.config(text="⏸️ Pause")

        # Setup optional video writer
        if self.save_var.get():
            out_name = f"annotated_{int(time.time())}.mp4"
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            self.video_writer = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.add_log_entry(f"Saving output to: {out_name}")

        self.worker_thread = threading.Thread(target=self.video_loop, args=(source,), daemon=True)
        self.worker_thread.start()

    def video_loop(self, source):
        is_camera = isinstance(source, int) or str(source).isdigit()
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else 0
        video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        if video_fps <= 0 or np.isnan(video_fps):
            video_fps = 25

        frame_count = 0
        prev_time = time.time()
        fps = 0.0

        while self.is_running:
            if self.is_paused:
                time.sleep(0.03)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            if is_camera:
                frame = cv2.flip(frame, 1)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            # Run Model Inference
            score, is_violent, mode_str = self.detector.process_frame(frame)

            # Log incident if violence begins
            if is_violent and (curr_time - self.last_incident_time > 2.0):
                self.incident_count += 1
                self.last_incident_time = curr_time
                self.root.after(0, self.add_log_entry, f"⚠️ VIOLENCE DETECTED ({score*100:.1f}%)", True)

            # Build progress label
            progress_str = None
            if total_frames > 0:
                pct = int((frame_count / total_frames) * 100)
                sec_curr = int(frame_count / video_fps)
                sec_tot = int(total_frames / video_fps)
                progress_str = f"{pct}% [{sec_curr//60:02d}:{sec_curr%60:02d}/{sec_tot//60:02d}:{sec_tot%60:02d}]"

            # Draw HUD
            annotated = draw_hud(frame, score, is_violent, mode_str, fps, progress_str)
            self.current_annotated_frame = annotated.copy()

            if self.video_writer:
                self.video_writer.write(annotated)

            # Convert for Tkinter Canvas
            self.root.after(0, self.render_frame, annotated)
            self.root.after(0, self.update_status_card, score, is_violent, fps, frame_count)

            # Maintain natural video speed for recorded files
            if not is_camera:
                time.sleep(max(0.005, (1.0 / video_fps) - 0.015))

        self.root.after(0, self.on_stream_end)

    def render_frame(self, frame_bgr):
        """Resizes frame maintaining aspect ratio and renders onto the canvas."""
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()

        if canvas_w <= 10 or canvas_h <= 10:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        fh, fw = frame_rgb.shape[:2]

        # Aspect ratio resize
        scale = min(canvas_w / fw, canvas_h / fh)
        nw, nh = int(fw * scale), int(fh * scale)

        resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(resized)
        self.tk_image = ImageTk.PhotoImage(image=img)

        self.video_canvas.delete("all")
        # Center image in canvas
        self.video_canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=self.tk_image)

    def on_stream_end(self):
        self.stop_stream()
        self.add_log_entry("Video playback completed.")
        self.draw_placeholder()

    def stop_stream(self):
        self.is_running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.btn_pause.config(text="⏸️ Pause")
        self.update_status_card(0.0, False, 0.0, 0)

    def on_close(self):
        self.stop_stream()
        self.root.destroy()


def launch():
    root = tk.Tk()
    app = ViolenceDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
