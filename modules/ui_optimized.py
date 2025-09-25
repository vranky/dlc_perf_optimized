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
from typing import Optional, Callable, Tuple
import platform
import queue

import modules.globals
from modules.performance_optimizer import FPSMonitor, PerformanceMetrics
from modules.video_capture import VideoCapturer
from modules.processors.frame.face_swapper_optimized import (
    get_optimized_face_swapper,
    process_frame_optimized
)
from modules.face_analyser import get_one_face
from modules.utilities import is_image, is_video
from PIL import ImageOps
import os

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

        # FPS tracking for display
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.display_fps = 0.0
        self.last_fps_update = time.time()

        # Create UI elements
        self.create_widgets()

        # Frame queues for async processing
        self.input_queue = queue.Queue(maxsize=5)
        self.output_queue = queue.Queue(maxsize=5)

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
        self.video_label = ctk.CTkLabel(
            self.main_frame, 
            text="Click 'Start Preview' to begin camera feed",
            width=640,
            height=480
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

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
            self.camera_started = False  # Reset camera started flag

            # Configure video capture
            width, height, fps = self.get_quality_settings()
            print(f"Starting video capture: {width}x{height} @ {fps}fps")
            success = self.video_capturer.start(width=width, height=height, fps=fps)
            
            if not success:
                print("Failed to start video capture")
                self.is_running = False
                return

            # Update button states
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)

            # Give camera a moment to initialize, then start capture
            print("Starting camera preview...")
            self.parent.after(500, self.simple_capture_test)

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

        try:
            ret, frame = self.video_capturer.read()
            if ret and frame is not None:
                # Apply frame skip
                skip = self.skip_var.get()
                if skip == 0 or (self.fps_monitor.metrics.processed_frames % (skip + 1) == 0):
                    # Add frame to processing queue (non-blocking)
                    try:
                        self.input_queue.put_nowait(frame)
                    except Exception:
                        pass  # Queue full, skip frame
        except Exception as e:
            print(f"Error in capture_frames: {e}")

        # Schedule next capture
        self.parent.after(1, self.capture_frames)

    def simple_capture_test(self):
        """Simple capture test with camera recovery"""
        if not self.is_running:
            return
        
        try:
            ret, frame = self.video_capturer.read()
            
            if ret and frame is not None:
                # Reset failure counter on success
                if hasattr(self, 'camera_failures'):
                    self.camera_failures = 0
                # Mark startup as complete after first successful frame
                if not hasattr(self, 'camera_started') or not self.camera_started:
                    print("✅ Camera started successfully!")
                    self.camera_started = True
                
                # Apply face swapping if source face is available
                if self.source_face is not None:
                    try:
                        processed_frame = process_frame_optimized(self.source_face, frame)
                        self.display_frame(processed_frame)
                    except Exception as e:
                        print(f"Face swap error: {e}")
                        # Fall back to raw frame
                        self.display_frame(frame)
                else:
                    # Just display raw frame
                    self.display_frame(frame)
                
                # Update FPS counter
                self.update_fps_counter()
                self.update_metrics()
                
            else:
                # Handle camera failure
                if not hasattr(self, 'camera_failures'):
                    self.camera_failures = 0
                if not hasattr(self, 'startup_attempts'):
                    self.startup_attempts = 0
                    
                self.camera_failures += 1
                self.startup_attempts += 1
                
                # Be more patient during startup (first 30 attempts)
                if self.startup_attempts <= 30:
                    if self.startup_attempts % 10 == 0:
                        print(f"Camera initializing... attempt {self.startup_attempts}/30")
                # After startup, be less patient  
                elif self.camera_failures >= 20:
                    print("Attempting camera recovery...")
                    try:
                        self.video_capturer.release()
                        # Wait a moment
                        self.parent.after(1000, self.recover_camera)
                        return
                    except Exception as e:
                        print(f"Camera recovery failed: {e}")
                
        except Exception as e:
            print(f"Error in simple_capture_test: {e}")
        
        # Schedule next capture
        if self.is_running:
            self.parent.after(33, self.simple_capture_test)  # ~30fps
    
    def recover_camera(self):
        """Attempt to recover camera connection"""
        if not self.is_running:
            return
            
        try:
            print("Reinitializing camera...")
            # Get camera ID from parent UI
            camera_id = 0  # Default
            if hasattr(self.parent, 'video_capturer'):
                camera_id = self.parent.video_capturer.device_index
            
            # Create new capturer
            from modules.video_capture import VideoCapturer
            self.video_capturer = VideoCapturer(camera_id)
            
            # Restart with current quality settings
            width, height, fps = self.get_quality_settings()
            success = self.video_capturer.start(width, height, fps)
            
            if success:
                print("Camera recovered successfully!")
                self.camera_failures = 0
                self.startup_attempts = 0  # Reset startup attempts
                self.camera_started = False  # Reset camera started flag
                # Resume capture
                self.parent.after(500, self.simple_capture_test)
            else:
                print("Camera recovery failed, stopping preview")
                self.stop_preview()
                
        except Exception as e:
            print(f"Camera recovery error: {e}")
            self.stop_preview()

    def show_test_image(self):
        """Show a test image to verify display is working"""
        try:
            print("DEBUG: Creating test image", flush=True)
            # Create a simple colored test image
            test_frame = np.full((480, 640, 3), [0, 255, 0], dtype=np.uint8)  # Green rectangle
            
            # Add some text
            cv2.putText(test_frame, "CAMERA TEST", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            print("DEBUG: Displaying test image", flush=True)
            self.display_frame(test_frame)
            print("DEBUG: Test image displayed", flush=True)
        except Exception as e:
            print(f"ERROR: Failed to show test image: {e}", flush=True)
            import traceback
            traceback.print_exc()

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

            # Convert to PIL Image and then to CTkImage
            image = Image.fromarray(frame_rgb)
            height, width = frame_rgb.shape[:2]
            ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(width, height))

            # Update label using CTkImage directly and force UI update
            self.video_label.configure(image=ctk_image, text="")
            self.video_label.image = ctk_image  # Keep reference
            
            # Force immediate UI update - this is crucial for CustomTkinter display
            self.parent.update()

        except Exception as e:
            print(f"Error displaying frame: {e}")
            import traceback
            traceback.print_exc()

    def update_fps_counter(self):
        """Update FPS counter with accurate display rate"""
        current_time = time.time()
        self.frame_count += 1
        
        # Update FPS every second
        if current_time - self.last_fps_update >= 1.0:
            time_elapsed = current_time - self.fps_start_time
            if time_elapsed > 0:
                self.display_fps = self.frame_count / time_elapsed
            
            # Reset counters every 5 seconds to keep the average current
            if current_time - self.fps_start_time >= 5.0:
                self.frame_count = 0
                self.fps_start_time = current_time
            
            self.last_fps_update = current_time

    def update_metrics(self):
        """Update performance metrics display"""
        # Update FPS with our calculated display FPS
        fps_text = f"FPS: {self.display_fps:.1f}"
        if self.display_fps < 20:
            fps_color = "red"
        elif self.display_fps < 25:
            fps_color = "orange"
        else:
            fps_color = "green"

        self.fps_label.configure(text=fps_text)

        # Calculate frame time from display FPS
        frame_time_ms = (1000.0 / self.display_fps) if self.display_fps > 0 else 0
        self.frame_time_label.configure(
            text=f"Frame Time: {frame_time_ms:.1f}ms"
        )

        # Update frame counter
        self.frames_label.configure(
            text=f"Frames: {self.frame_count}"
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
        
        # Image labels for preview
        self.source_label = None
        self.target_label = None
        
        # Recent directories for file dialogs
        self.recent_directory_source = None
        self.recent_directory_target = None

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

        # File selection frame
        file_frame = ctk.CTkFrame(self.root)
        file_frame.pack(fill=tk.X, padx=10, pady=10)

        # Source image section
        source_frame = ctk.CTkFrame(file_frame)
        source_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        ctk.CTkLabel(source_frame, text="Source Image", font=("Arial", 14, "bold")).pack(pady=5)

        self.source_label = ctk.CTkLabel(source_frame, text="No image selected", width=200, height=150)
        self.source_label.pack(pady=5)

        select_source_btn = ctk.CTkButton(
            source_frame,
            text="Select Face Image",
            command=self.select_source_path
        )
        select_source_btn.pack(pady=5)

        # Target image section  
        target_frame = ctk.CTkFrame(file_frame)
        target_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        ctk.CTkLabel(target_frame, text="Target Image/Video", font=("Arial", 14, "bold")).pack(pady=5)

        self.target_label = ctk.CTkLabel(target_frame, text="No target selected", width=200, height=150)
        self.target_label.pack(pady=5)

        select_target_btn = ctk.CTkButton(
            target_frame,
            text="Select Target",
            command=self.select_target_path
        )
        select_target_btn.pack(pady=5)

        # Camera selection
        camera_frame = ctk.CTkFrame(self.root)
        camera_frame.pack(fill=tk.X, padx=10, pady=5)

        ctk.CTkLabel(
            camera_frame,
            text="Camera:"
        ).pack(side=tk.LEFT, padx=5)

        self.camera_var = tk.IntVar(value=1)
        self.camera_menu = ctk.CTkOptionMenu(
            camera_frame,
            values=["Camera 1", "Camera 2"],
            command=self.on_camera_changed
        )
        self.camera_menu.set("Camera 1")  # Set default display value
        self.camera_menu.pack(side=tk.LEFT, padx=5)

        # Initialize video capturer - use Camera 1 as default (Camera 0 doesn't work on this system)
        self.video_capturer = VideoCapturer(1)

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

    def select_source_path(self):
        """Select source face image"""
        img_ft = ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")
        
        source_path = ctk.filedialog.askopenfilename(
            title="Select a face image",
            initialdir=self.recent_directory_source,
            filetypes=[img_ft]
        )
        
        if is_image(source_path):
            modules.globals.source_path = source_path
            self.recent_directory_source = os.path.dirname(source_path)
            
            # Update preview
            image = self.render_image_preview(source_path, (180, 120))
            self.source_label.configure(image=image, text="")
            
            # Update source face for preview
            if self.preview:
                source_img = cv2.imread(source_path)
                if source_img is not None:
                    self.preview.source_face = get_one_face(source_img)
        else:
            modules.globals.source_path = None
            self.source_label.configure(image=None, text="No image selected")

    def select_target_path(self):
        """Select target image or video"""
        img_ft = ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")
        vid_ft = ("Video files", "*.mp4 *.avi *.mov *.mkv")
        
        target_path = ctk.filedialog.askopenfilename(
            title="Select target image or video",
            initialdir=self.recent_directory_target,
            filetypes=[img_ft, vid_ft]
        )
        
        if is_image(target_path):
            modules.globals.target_path = target_path
            self.recent_directory_target = os.path.dirname(target_path)
            
            # Update preview
            image = self.render_image_preview(target_path, (180, 120))
            self.target_label.configure(image=image, text="")
            
        elif is_video(target_path):
            modules.globals.target_path = target_path
            self.recent_directory_target = os.path.dirname(target_path)
            
            # Update preview with first frame
            video_frame = self.render_video_preview(target_path, (180, 120))
            self.target_label.configure(image=video_frame, text="")
        else:
            modules.globals.target_path = None
            self.target_label.configure(image=None, text="No target selected")

    def render_image_preview(self, image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
        """Render image preview"""
        image = Image.open(image_path)
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)

    def render_video_preview(self, video_path: str, size: Tuple[int, int], frame_number: int = 0) -> ctk.CTkImage:
        """Render video preview from first frame"""
        capture = cv2.VideoCapture(video_path)
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if has_frame:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if size:
                image = ImageOps.fit(image, size, Image.LANCZOS)
            capture.release()
            return ctk.CTkImage(image, size=image.size)
        capture.release()
        return None

    def run(self):
        """Run the UI"""
        self.root.mainloop()


def create_optimized_ui():
    """Create and return optimized UI instance"""
    return OptimizedUI()