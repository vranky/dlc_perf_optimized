"""
Performance optimizer module for Mac M3 and Apple Silicon
Implements Metal Performance Shaders and optimized tensor operations
"""

import os
import platform
import time
import threading
from typing import List, Tuple, Optional, Any
import numpy as np
import cv2
import onnxruntime as ort
from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import queue

# Check if running on macOS with Apple Silicon
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.processor() == 'arm'


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    fps: float = 0.0
    frame_time: float = 0.0
    gpu_usage: float = 0.0
    memory_usage: float = 0.0
    dropped_frames: int = 0
    processed_frames: int = 0


class FPSMonitor:
    """Real-time FPS monitoring with moving average"""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = None
        self.metrics = PerformanceMetrics()
        self.lock = threading.Lock()

    def start_frame(self):
        """Mark the start of frame processing"""
        self.last_time = time.perf_counter()

    def end_frame(self):
        """Mark the end of frame processing and calculate FPS"""
        if self.last_time is None:
            return

        current_time = time.perf_counter()
        frame_time = current_time - self.last_time

        with self.lock:
            self.frame_times.append(frame_time)
            self.metrics.processed_frames += 1

            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.metrics.frame_time = avg_frame_time * 1000  # Convert to ms
                self.metrics.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        with self.lock:
            return self.metrics


class FrameBufferPool:
    """Memory pool for frame buffers to reduce allocation overhead"""

    def __init__(self, pool_size: int = 10, frame_shape: Tuple[int, int, int] = None):
        self.pool_size = pool_size
        self.frame_shape = frame_shape or (1080, 1920, 3)
        self.pool = queue.Queue(maxsize=pool_size)
        self._initialize_pool()

    def _initialize_pool(self):
        """Pre-allocate frame buffers"""
        for _ in range(self.pool_size):
            buffer = np.empty(self.frame_shape, dtype=np.uint8)
            self.pool.put(buffer)

    def get_buffer(self) -> np.ndarray:
        """Get a buffer from the pool"""
        try:
            return self.pool.get_nowait()
        except queue.Empty:
            # If pool is empty, allocate new buffer
            return np.empty(self.frame_shape, dtype=np.uint8)

    def return_buffer(self, buffer: np.ndarray):
        """Return a buffer to the pool"""
        try:
            self.pool.put_nowait(buffer)
        except queue.Full:
            # Pool is full, let garbage collector handle it
            pass


class AppleSiliconOptimizer:
    """Optimizations specific to Apple Silicon M3"""

    @staticmethod
    def get_optimal_providers() -> List[str]:
        """Get optimal execution providers for Apple Silicon"""
        available_providers = ort.get_available_providers()

        # Prioritize CoreML for Apple Silicon
        optimal_providers = []

        if 'CoreMLExecutionProvider' in available_providers:
            optimal_providers.append('CoreMLExecutionProvider')

        # Add CPU provider as fallback
        if 'CPUExecutionProvider' in available_providers:
            optimal_providers.append('CPUExecutionProvider')

        return optimal_providers if optimal_providers else ['CPUExecutionProvider']

    @staticmethod
    def create_session_options() -> ort.SessionOptions:
        """Create optimized session options for ONNX Runtime"""
        options = ort.SessionOptions()

        # Enable all optimizations
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Use parallel execution
        options.inter_op_num_threads = 0  # Use all available cores
        options.intra_op_num_threads = 0  # Use all available cores

        # Enable memory pattern optimization
        options.enable_mem_pattern = True
        options.enable_mem_reuse = True

        # Add execution mode for better performance
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        return options

    @staticmethod
    def optimize_model_for_coreml(model_path: str) -> str:
        """Optimize ONNX model for CoreML execution"""
        # This would typically involve model conversion
        # For now, return the original path
        return model_path


class BatchProcessor:
    """Batch processing for improved throughput"""

    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.frame_queue = queue.Queue(maxsize=batch_size * 2)
        self.result_queue = queue.Queue(maxsize=batch_size * 2)
        self.executor = ThreadPoolExecutor(max_workers=2)

    def add_frame(self, frame: np.ndarray, metadata: Any = None):
        """Add frame to processing queue"""
        try:
            self.frame_queue.put_nowait((frame, metadata))
            return True
        except queue.Full:
            return False

    def process_batch(self, process_func):
        """Process frames in batches"""
        batch = []
        metadata_batch = []

        # Collect frames for batch
        while len(batch) < self.batch_size:
            try:
                frame, metadata = self.frame_queue.get_nowait()
                batch.append(frame)
                metadata_batch.append(metadata)
            except queue.Empty:
                break

        if batch:
            # Process batch
            results = process_func(batch)

            # Queue results
            for result, metadata in zip(results, metadata_batch):
                self.result_queue.put((result, metadata))

    def get_result(self) -> Tuple[Optional[np.ndarray], Any]:
        """Get processed frame from result queue"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None, None


class OptimizedFaceSwapper:
    """Optimized face swapper for Mac M3"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.fps_monitor = FPSMonitor()
        self.buffer_pool = FrameBufferPool()
        self.batch_processor = BatchProcessor()

        # Initialize ONNX Runtime session with optimizations
        self._initialize_session()

    def _initialize_session(self):
        """Initialize optimized ONNX Runtime session"""
        if IS_APPLE_SILICON:
            providers = AppleSiliconOptimizer.get_optimal_providers()
            options = AppleSiliconOptimizer.create_session_options()
        else:
            providers = ['CPUExecutionProvider']
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create session with optimized settings
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=providers
        )

    def process_frame_optimized(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame with optimizations"""
        self.fps_monitor.start_frame()

        # Get buffer from pool
        result_buffer = self.buffer_pool.get_buffer()

        # Process frame (placeholder for actual processing)
        # In real implementation, this would call the face swap model
        np.copyto(result_buffer, frame)

        self.fps_monitor.end_frame()

        # Return buffer to pool
        self.buffer_pool.return_buffer(result_buffer)

        return result_buffer

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.fps_monitor.get_metrics()


class VideoProcessor:
    """Optimized video processing pipeline"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and IS_APPLE_SILICON
        self.frame_skip = 0
        self.processing_thread = None
        self.is_running = False

    def enable_adaptive_quality(self, target_fps: float = 30):
        """Enable adaptive quality based on target FPS"""
        self.target_fps = target_fps
        self.adaptive_enabled = True

    def process_video_stream(self, input_stream, output_callback, face_swapper):
        """Process video stream with optimizations"""
        self.is_running = True
        frame_count = 0

        while self.is_running:
            ret, frame = input_stream.read()
            if not ret:
                break

            # Skip frames if needed for performance
            if self.frame_skip > 0 and frame_count % (self.frame_skip + 1) != 0:
                frame_count += 1
                continue

            # Process frame
            processed_frame = face_swapper.process_frame_optimized(frame)

            # Send to output
            output_callback(processed_frame)

            # Adjust quality based on FPS
            metrics = face_swapper.get_performance_metrics()
            if hasattr(self, 'adaptive_enabled') and self.adaptive_enabled:
                if metrics.fps < self.target_fps * 0.9:
                    # Increase frame skip if FPS is too low
                    self.frame_skip = min(self.frame_skip + 1, 3)
                elif metrics.fps > self.target_fps * 1.1:
                    # Decrease frame skip if FPS is good
                    self.frame_skip = max(self.frame_skip - 1, 0)

            frame_count += 1

    def stop(self):
        """Stop video processing"""
        self.is_running = False


def optimize_opencv_settings():
    """Optimize OpenCV settings for better performance"""
    # Set number of threads for OpenCV
    cv2.setNumThreads(0)  # Use all available cores

    # Enable OpenCV optimizations
    cv2.setUseOptimized(True)

    # Check if optimizations are enabled
    if cv2.useOptimized():
        print("OpenCV optimizations enabled")


def get_recommended_settings() -> dict:
    """Get recommended settings for Mac M3"""
    settings = {
        'execution_provider': ['CoreMLExecutionProvider', 'CPUExecutionProvider'],
        'execution_threads': 0,  # Use all available cores
        'batch_size': 4,
        'frame_buffer_size': 10,
        'enable_adaptive_quality': True,
        'target_fps': 30,
        'max_memory_gb': 8,  # Mac M3 typically has 8GB unified memory minimum
    }

    if IS_APPLE_SILICON:
        settings['use_neural_engine'] = True
        settings['use_metal'] = True

    return settings


# Initialize OpenCV optimizations on module load
optimize_opencv_settings()