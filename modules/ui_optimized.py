"""
Optimized UI module with real-time FPS monitoring for Mac M3
"""

import time
import threading
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Callable
import platform

import modules.globals
from modules.performance_optimizer import FPSMonitor, PerformanceMetrics
from modules.video_capture import VideoCapturer
from modules.processors.frame.face_swapper_optimized import (
    get_optimized_face_swapper,
    process_frame_optimized
)
from modules.face_analyser import get_one_face

# Check if running on Apple Silicon
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.processor() == 'arm'


class OptimizedPreviewWindow:
    """Optimized preview window with FPS monitoring"""

    def __init__(self, parent, video_capturer: VideoCapturer):
        self.parent = parent
        self.video_capturer = video_capturer
        self.fps_monitor = FPSMonitor(window_size=30)
        self.is_running = False
        self.process_thread = None
        self.display_thread = None

        # Create UI elements
        self.create_widgets()

        # Frame queues for async processing
        self.input_queue = threading.Queue(maxsize=5)
        self.output_queue = threading.Queue(maxsize=5)

        # Source face for swapping
        self.source_face = None
        if modules.globals.source_path:
            source_img = cv2.imread(modules.globals.source_path)
            if source_img is not None:
                self.source_face = get_one_face(source_img)

    def create_widgets(self):
        """Create optimized preview widgets"""
        # Main frame
        self.main_frame = ctk.CTkFrame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video preview
        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Performance metrics frame
        self.metrics_frame = ctk.CTkFrame(self.main_frame)
        self.metrics_frame.pack(fill=tk.X, pady=5)

        # FPS display
        self.fps_label = ctk.CTkLabel(
            self.metrics_frame,
            text="FPS: 0.0",
            font=("Arial", 14, "bold")
        )
        self.fps_label.pack(side=tk.LEFT, padx=10)

        # Frame time display
        self.frame_time_label = ctk.CTkLabel(
            self.metrics_frame,
            text="Frame Time: 0.0ms",
            font=("Arial", 12)
        )
        self.frame_time_label.pack(side=tk.LEFT, padx=10)

        # Processed frames counter
        self.frames_label = ctk.CTkLabel(
            self.metrics_frame,
            text="Frames: 0",
            font=("Arial", 12)
        )
        self.frames_label.pack(side=tk.LEFT, padx=10)

        # Control buttons
        self.control_frame = ctk.CTkFrame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=5)

        self.start_button = ctk.CTkButton(
            self.control_frame,
            text="Start Preview",
            command=self.start_preview
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ctk.CTkButton(
            self.control_frame,
            text="Stop Preview",
            command=self.stop_preview,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Quality settings
        self.quality_frame = ctk.CTkFrame(self.main_frame)
        self.quality_frame.pack(fill=tk.X, pady=5)

        ctk.CTkLabel(
            self.quality_frame,
            text="Quality:"
        ).pack(side=tk.LEFT, padx=5)

        self.quality_var = tk.StringVar(value="balanced")
        self.quality_menu = ctk.CTkOptionMenu(
            self.quality_frame,
            values=["performance", "balanced", "quality"],
            variable=self.quality_var,
            command=self.on_quality_changed
        )
        self.quality_menu.pack(side=tk.LEFT, padx=5)

        # Frame skip setting
        ctk.CTkLabel(
            self.quality_frame,
            text="Frame Skip:"
        ).pack(side=tk.LEFT, padx=5)

        self.skip_var = tk.IntVar(value=0)
        self.skip_slider = ctk.CTkSlider(
            self.quality_frame,
            from_=0,
            to=3,
            number_of_steps=3,
            variable=self.skip_var
        )
        self.skip_slider.pack(side=tk.LEFT, padx=5)

    def start_preview(self):
        """Start optimized preview"""
        if not self.is_running:
            self.is_running = True

            # Configure video capture
            width, height, fps = self.get_quality_settings()
            self.video_capturer.start(width=width, height=height, fps=fps)

            # Update button states
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)

            # Start processing threads
            self.process_thread = threading.Thread(
                target=self.process_frames_worker,
                daemon=True
            )
            self.process_thread.start()

            self.display_thread = threading.Thread(
                target=self.display_frames_worker,
                daemon=True
            )
            self.display_thread.start()

            # Start capture
            self.capture_frames()

    def stop_preview(self):
        """Stop preview"""
        self.is_running = False

        # Stop video capture
        if self.video_capturer:
            self.video_capturer.release()

        # Update button states
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)

        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except:
                pass

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except:
                pass

    def capture_frames(self):
        """Capture frames from camera"""
        if not self.is_running:
            return

        ret, frame = self.video_capturer.read()
        if ret and frame is not None:
            # Apply frame skip
            skip = self.skip_var.get()
            if skip == 0 or (self.fps_monitor.metrics.processed_frames % (skip + 1) == 0):
                # Add frame to processing queue (non-blocking)
                try:
                    self.input_queue.put_nowait(frame)
                except:
                    pass  # Queue full, skip frame

        # Schedule next capture
        self.parent.after(1, self.capture_frames)

    def process_frames_worker(self):
        """Worker thread for frame processing"""
        swapper = get_optimized_face_swapper()

        while self.is_running:
            try:
                # Get frame from input queue
                frame = self.input_queue.get(timeout=0.1)

                # Start FPS monitoring
                self.fps_monitor.start_frame()

                # Process frame
                if self.source_face is not None:
                    processed = process_frame_optimized(self.source_face, frame)
                else:
                    processed = frame

                # End FPS monitoring
                self.fps_monitor.end_frame()

                # Add to output queue
                try:
                    self.output_queue.put_nowait(processed)
                except:
                    pass  # Queue full, drop frame

            except:
                continue  # Queue empty or timeout

    def display_frames_worker(self):
        """Worker thread for displaying frames"""
        while self.is_running:
            try:
                # Get processed frame
                frame = self.output_queue.get(timeout=0.1)

                # Convert and display
                self.display_frame(frame)

                # Update metrics
                self.update_metrics()

            except:
                continue  # Queue empty or timeout

    def display_frame(self, frame: np.ndarray):
        """Display frame in UI"""
        try:
            # Mirror if needed
            if modules.globals.live_mirror:
                frame = cv2.flip(frame, 1)

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize to fit display
            height, width = frame_rgb.shape[:2]
            max_width = 800
            max_height = 600

            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))

            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=image)

            # Update label
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # Keep reference

        except Exception as e:
            print(f"Error displaying frame: {e}")

    def update_metrics(self):
        """Update performance metrics display"""
        metrics = self.fps_monitor.get_metrics()

        # Update FPS
        fps_text = f"FPS: {metrics.fps:.1f}"
        if metrics.fps < 20:
            fps_color = "red"
        elif metrics.fps < 25:
            fps_color = "orange"
        else:
            fps_color = "green"

        self.fps_label.configure(text=fps_text)

        # Update frame time
        self.frame_time_label.configure(
            text=f"Frame Time: {metrics.frame_time:.1f}ms"
        )

        # Update frame counter
        self.frames_label.configure(
            text=f"Frames: {metrics.processed_frames}"
        )

    def get_quality_settings(self) -> Tuple[int, int, int]:
        """Get resolution and FPS based on quality setting"""
        quality = self.quality_var.get()

        if quality == "performance":
            # Lower resolution for better performance
            return 640, 480, 60
        elif quality == "balanced":
            # Balanced settings
            return 960, 540, 30
        else:  # quality
            # Higher quality
            return 1280, 720, 30

    def on_quality_changed(self, choice):
        """Handle quality setting change"""
        if self.is_running:
            # Restart preview with new settings
            self.stop_preview()
            self.parent.after(100, self.start_preview)


class OptimizedUI:
    """Main optimized UI class"""

    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("DLC Optimized - Mac M3 Performance")
        self.root.geometry("1000x700")

        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Video capturer
        self.video_capturer = None

        # Create UI
        self.create_ui()

    def create_ui(self):
        """Create main UI"""
        # Title
        title_label = ctk.CTkLabel(
            self.root,
            text="Deep Live Cam - Optimized for Mac M3",
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=10)

        # System info
        if IS_APPLE_SILICON:
            info_text = "Apple Silicon Detected - Optimizations Enabled"
        else:
            info_text = "Standard CPU Mode"

        info_label = ctk.CTkLabel(
            self.root,
            text=info_text,
            font=("Arial", 12)
        )
        info_label.pack()

        # Camera selection
        camera_frame = ctk.CTkFrame(self.root)
        camera_frame.pack(fill=tk.X, padx=10, pady=5)

        ctk.CTkLabel(
            camera_frame,
            text="Camera:"
        ).pack(side=tk.LEFT, padx=5)

        self.camera_var = tk.IntVar(value=0)
        self.camera_menu = ctk.CTkOptionMenu(
            camera_frame,
            values=["Camera 0", "Camera 1", "Camera 2"],
            command=self.on_camera_changed
        )
        self.camera_menu.pack(side=tk.LEFT, padx=5)

        # Initialize video capturer
        self.video_capturer = VideoCapturer(0)

        # Preview window
        self.preview = OptimizedPreviewWindow(self.root, self.video_capturer)

    def on_camera_changed(self, choice):
        """Handle camera selection change"""
        camera_id = int(choice.split()[-1])

        # Stop current capture
        if self.video_capturer:
            self.video_capturer.release()

        # Create new capturer
        self.video_capturer = VideoCapturer(camera_id)

        # Update preview
        self.preview.video_capturer = self.video_capturer

    def run(self):
        """Run the UI"""
        self.root.mainloop()


def create_optimized_ui():
    """Create and return optimized UI instance"""
    return OptimizedUI()