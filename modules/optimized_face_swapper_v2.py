"""
Ultra-optimized Face Swapper for Mac M1/M3 - Memory Efficient Implementation
Based on research from Deep-Live-Cam, DeepFaceLive, and Apple CoreML best practices
Target: 20+ FPS on Mac M1/M3
"""

import os
import cv2
import numpy as np
import threading
import time
import queue
import platform
import gc
import psutil
from typing import Optional, Tuple, Dict, Any
from collections import deque
from dataclasses import dataclass

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    print("Warning: insightface not available")
    INSIGHTFACE_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("Warning: onnxruntime not available")
    ONNX_AVAILABLE = False

import modules.globals
from modules.face_types import Face

# Memory and Performance Constants
MAX_MEMORY_GB = 8  # Target memory usage limit
FACE_CACHE_SIZE = 50  # Reduced cache size for memory efficiency
FRAME_BUFFER_POOL_SIZE = 6  # Smaller pool for memory efficiency
FACE_DETECTION_INTERVAL = 5  # Detect face every 5 frames for smooth tracking

# Apple Silicon Detection
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.processor() == 'arm'

@dataclass
class PerformanceMetrics:
    fps: float = 0.0
    memory_mb: float = 0.0
    thermal_state: str = "normal"
    model_load_time: float = 0.0

class MemoryManager:
    """Ultra-efficient memory manager for Apple Silicon"""
    
    def __init__(self, max_memory_gb: float = MAX_MEMORY_GB):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
        
    def should_gc(self) -> bool:
        """Check if garbage collection is needed"""
        current_mb = self.get_memory_usage()
        return current_mb > (self.max_memory_bytes / 1024 / 1024) * 0.8
        
    def force_cleanup(self):
        """Force memory cleanup"""
        if self.should_gc():
            gc.collect()

class OptimizedFrameBuffer:
    """Memory-efficient frame buffer using pre-allocated numpy arrays"""
    
    def __init__(self, size: int = FRAME_BUFFER_POOL_SIZE, frame_shape: Tuple[int, int, int] = (480, 640, 3)):
        self.size = size
        self.frame_shape = frame_shape
        self.buffers = queue.Queue(maxsize=size)
        self._initialize_buffers()
        
    def _initialize_buffers(self):
        """Pre-allocate frame buffers to avoid runtime allocation"""
        for _ in range(self.size):
            buffer = np.zeros(self.frame_shape, dtype=np.uint8)
            self.buffers.put(buffer)
            
    def get_buffer(self) -> np.ndarray:
        """Get a pre-allocated buffer"""
        try:
            return self.buffers.get_nowait()
        except queue.Empty:
            # Emergency allocation if pool exhausted
            return np.zeros(self.frame_shape, dtype=np.uint8)
            
    def return_buffer(self, buffer: np.ndarray):
        """Return buffer to pool"""
        try:
            self.buffers.put_nowait(buffer)
        except queue.Full:
            # Pool is full, let buffer be garbage collected
            pass

class UltraOptimizedFaceSwapper:
    """Ultra-optimized face swapper for 20+ FPS on Mac M1/M3"""
    
    def __init__(self):
        self.model = None
        self.face_analysis = None
        self.memory_manager = MemoryManager()
        self.frame_buffer = OptimizedFrameBuffer()
        self.metrics = PerformanceMetrics()
        
        # Face detection cache
        self.cached_target_face = None
        self.face_cache_timestamp = 0
        self.frame_count = 0
        
        # Performance monitoring
        self.fps_times = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        # Thread safety
        self.model_lock = threading.RLock()
        
    def initialize(self) -> bool:
        """Initialize models with optimal Apple Silicon configuration"""
        if not INSIGHTFACE_AVAILABLE or not ONNX_AVAILABLE:
            print("Required libraries not available")
            return False
            
        start_time = time.time()
        
        try:
            # Configure ONNX Runtime for Apple Silicon
            providers, provider_options = self._get_optimal_providers()
            
            # Initialize face analysis with optimized settings
            # Note: FaceAnalysis might not support provider_options parameter
            try:
                self.face_analysis = FaceAnalysis(
                    providers=providers,
                    allowed_modules=['detection', 'recognition']
                )
            except TypeError:
                # Fallback if provider_options not supported
                self.face_analysis = FaceAnalysis(
                    providers=['CPUExecutionProvider'],  # Safe fallback
                    allowed_modules=['detection', 'recognition']
                )
            
            # Set optimal context for Apple Silicon
            if IS_APPLE_SILICON:
                self.face_analysis.prepare(ctx_id=-1, det_size=(320, 320))  # Smaller detection size for speed
            else:
                self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
            
            # Load face swapping model
            model_path = self._get_model_path()
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                return False
                
            # Create optimized ONNX session
            session_options = self._create_session_options()
            self.model = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers,
                provider_options=provider_options
            )
            
            self.metrics.model_load_time = time.time() - start_time
            print(f"Model initialized in {self.metrics.model_load_time:.2f}s")
            print(f"Providers: {providers}")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize face swapper: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_optimal_providers(self) -> Tuple[list, list]:
        """Get optimal ONNX providers for Apple Silicon"""
        if not IS_APPLE_SILICON:
            return ['CPUExecutionProvider'], [{}]
            
        providers = []
        provider_options = []
        
        # Primary: CoreML for Neural Engine utilization
        if 'CoreMLExecutionProvider' in ort.get_available_providers():
            providers.append('CoreMLExecutionProvider')
            provider_options.append({
                'MLComputeUnits': 'ALL',  # CPU, GPU, Neural Engine
                'ModelFormat': 'MLProgram',
                'RequireStaticInputShapes': '0',
                'AllowLowPrecisionAccumulationOnGPU': '1',
                'EnableOnSubgraph': '1'
            })
        
        # Fallback: Optimized CPU
        providers.append('CPUExecutionProvider')
        provider_options.append({
            'intra_op_num_threads': 8,  # M1/M3 performance cores
            'inter_op_num_threads': 4,
            'omp_num_threads': 8
        })
        
        return providers, provider_options
    
    def _create_session_options(self) -> ort.SessionOptions:
        """Create optimized session options"""
        options = ort.SessionOptions()
        
        # Enable all graph optimizations
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Optimize for throughput on Apple Silicon
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        options.inter_op_num_threads = 4
        options.intra_op_num_threads = 8
        
        # Memory optimizations
        options.enable_mem_pattern = True
        options.enable_mem_reuse = True
        
        # Disable verbose logging for performance
        options.log_severity_level = 3
        
        return options
    
    def _get_model_path(self) -> str:
        """Get path to face swapping model"""
        # Try multiple possible locations for the model
        possible_paths = [
            # Direct path from current working directory
            "models/inswapper_128_fp16.onnx",
            # Relative to this file
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "inswapper_128_fp16.onnx"),
            # Absolute path construction
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "inswapper_128_fp16.onnx"),
            # Default insightface models location
            os.path.expanduser("~/.insightface/models/inswapper_128_fp16.onnx"),
        ]
        
        for path in possible_paths:
            normalized_path = os.path.normpath(path)
            print(f"Checking model path: {normalized_path}")
            if os.path.exists(normalized_path):
                print(f"✅ Found model at: {normalized_path}")
                return normalized_path
        
        print(f"❌ Model not found in any of the checked paths")
        # If no model found, return the first path for error reporting
        return os.path.normpath(possible_paths[0])
    
    def get_cached_face(self, frame: np.ndarray) -> Optional[Face]:
        """Get cached face or detect new one with reduced frequency"""
        current_time = time.time()
        self.frame_count += 1
        
        # More frequent detection for smooth tracking
        should_detect = (
            self.cached_target_face is None or
            self.frame_count % FACE_DETECTION_INTERVAL == 0 or
            current_time - self.face_cache_timestamp > 0.5  # 500ms timeout for smooth tracking
        )
        
        if should_detect:
            try:
                # Use optimized face analysis
                faces = self.face_analysis.get(frame)
                if faces and len(faces) > 0:
                    # Use the most confident face
                    best_face = max(faces, key=lambda x: x.det_score)
                    self.cached_target_face = best_face
                    self.face_cache_timestamp = current_time
                    
                    # Memory cleanup check
                    if self.frame_count % 100 == 0:
                        self.memory_manager.force_cleanup()
                        
            except Exception as e:
                print(f"Face detection error: {e}")
        
        return self.cached_target_face
    
    def swap_face(self, source_face: Face, target_face: Face, frame: np.ndarray) -> np.ndarray:
        """Perform optimized face swapping"""
        if not self.model or not source_face or not target_face:
            return frame
            
        try:
            with self.model_lock:
                # Get buffer from pool
                output_buffer = self.frame_buffer.get_buffer()
                
                # Prepare model inputs
                input_data = self._prepare_model_inputs(source_face, target_face, frame)
                
                # Run inference
                outputs = self.model.run(None, input_data)
                
                # Process outputs
                result = self._process_model_outputs(outputs[0], frame, target_face)
                
                # Copy result to output buffer
                if result.shape == output_buffer.shape:
                    np.copyto(output_buffer, result)
                else:
                    output_buffer = result.copy()
                
                # Return buffer to pool
                self.frame_buffer.return_buffer(output_buffer)
                
                return result
                
        except Exception as e:
            print(f"Face swap error: {e}")
            return frame
    
    def _prepare_model_inputs(self, source_face: Face, target_face: Face, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Prepare model inputs for inswapper model"""
        inputs = {}
        
        # For inswapper model, we need:
        # 1. Target face region (128x128)  
        # 2. Source face embedding (512-dim vector)
        
        # Extract target face region
        bbox = target_face.bbox.astype(np.int32)
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            # Extract and resize face region
            face_region = frame[y1:y2, x1:x2]
            face_region = cv2.resize(face_region, (128, 128))
            
            # Normalize to [-1, 1] range (standard for inswapper)
            face_region = face_region.astype(np.float32) / 127.5 - 1.0
            
            # Convert to CHW format and add batch dimension
            face_region = face_region.transpose(2, 0, 1)
            inputs['target'] = np.expand_dims(face_region, axis=0)
        else:
            # Create dummy input if bbox is invalid
            dummy_face = np.zeros((1, 3, 128, 128), dtype=np.float32)
            inputs['target'] = dummy_face
        
        # Source face embedding
        if hasattr(source_face, 'normed_embedding') and source_face.normed_embedding is not None:
            embedding = source_face.normed_embedding.astype(np.float32)
            inputs['source'] = np.expand_dims(embedding, axis=0)
        elif hasattr(source_face, 'embedding') and source_face.embedding is not None:
            embedding = source_face.embedding.astype(np.float32)
            # Normalize embedding if needed
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            inputs['source'] = np.expand_dims(embedding, axis=0)
        else:
            # Create dummy embedding if not available
            inputs['source'] = np.zeros((1, 512), dtype=np.float32)
        
        return inputs
    
    def _process_model_outputs(self, output: np.ndarray, original_frame: np.ndarray, target_face: Face) -> np.ndarray:
        """Process inswapper model outputs efficiently"""
        try:
            # Inswapper output is typically (1, 3, 128, 128) - a swapped face
            if output.ndim == 4:  # Remove batch dimension
                output = output[0]
            
            if output.shape[0] == 3:  # CHW to HWC format
                output = output.transpose(1, 2, 0)
            
            # Convert from [-1, 1] range back to [0, 255]
            output = ((output + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            
            # Get target face bbox
            bbox = target_face.bbox.astype(np.int32)
            x1, y1, x2, y2 = bbox
            
            # Ensure bbox is within frame bounds
            h, w = original_frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face_width = x2 - x1
            face_height = y2 - y1
            
            if face_width > 0 and face_height > 0:
                # Resize swapped face to match target face size
                swapped_face = cv2.resize(output, (face_width, face_height))
                
                # Create result frame
                result = original_frame.copy()
                
                # Simple paste (could be improved with blending)
                result[y1:y2, x1:x2] = swapped_face
                
                return result
            
        except Exception as e:
            print(f"Output processing error: {e}")
            import traceback
            traceback.print_exc()
        
        return original_frame
    
    def process_frame(self, source_face: Face, frame: np.ndarray) -> np.ndarray:
        """Main frame processing entry point"""
        start_time = time.time()
        
        # Get target face from cache
        target_face = self.get_cached_face(frame)
        
        if not target_face:
            return frame
            
        # Perform face swap
        result = self.swap_face(source_face, target_face, frame)
        
        # Update FPS metrics
        self._update_fps_metrics(start_time)
        
        return result
    
    def _update_fps_metrics(self, start_time: float):
        """Update FPS and memory metrics"""
        frame_time = time.time() - start_time
        self.fps_times.append(frame_time)
        
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            # Calculate FPS
            if self.fps_times:
                avg_frame_time = sum(self.fps_times) / len(self.fps_times)
                self.metrics.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Update memory usage
            self.metrics.memory_mb = self.memory_manager.get_memory_usage()
            
            self.last_fps_time = current_time
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.metrics
    
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
        if self.face_analysis:
            del self.face_analysis
        self.memory_manager.force_cleanup()

# Global instance
_face_swapper_instance = None
_instance_lock = threading.Lock()

def get_face_swapper() -> UltraOptimizedFaceSwapper:
    """Get singleton face swapper instance"""
    global _face_swapper_instance
    
    with _instance_lock:
        if _face_swapper_instance is None:
            _face_swapper_instance = UltraOptimizedFaceSwapper()
            if not _face_swapper_instance.initialize():
                raise RuntimeError("Failed to initialize face swapper")
        
        return _face_swapper_instance

def process_frame_ultra_optimized(source_face: Face, frame: np.ndarray) -> np.ndarray:
    """Process frame with ultra optimization"""
    swapper = get_face_swapper()
    return swapper.process_frame(source_face, frame)