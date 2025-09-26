"""
Ultra-Fast Face Swapper for Real-time Performance
Uses simplified approach for 20+ FPS on Mac M1/M3
"""

import cv2
import numpy as np
import time
from typing import Optional
import threading

from modules.face_types import Face

# Performance tracking utilities

class FastFaceSwapper:
    """Ultra-fast face swapper using template matching and blending"""
    
    def __init__(self):
        self.face_swapper_model = None
        self.cache_lock = threading.Lock()
        
        # Face detection caching - optimized for expressions vs performance balance
        self.cached_target_face = None
        self.face_cache_time = 0
        self.face_detection_interval = 0.2  # Detect face every 200ms for better expression tracking
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0
        
        # High-performance caching
        self.source_face_aligned = None
        self.cached_alignment_matrix = None
        self.cached_swap_result = None
        self.last_target_bbox = None
        self.alignment_cache_time = 0
        self.use_neural_swap = False  # Start with template-based for speed
        
        # Initialize the proper InsightFace face swapper
        self._initialize_face_swapper()
    
    def _initialize_face_swapper(self):
        """Initialize InsightFace face swapper with performance optimizations"""
        try:
            import insightface
            import modules.globals
            import os
            
            # Find the model path
            model_paths = [
                "models/inswapper_128_fp16.onnx",
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "inswapper_128_fp16.onnx"),
                os.path.expanduser("~/.insightface/models/inswapper_128_fp16.onnx")
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                print("❌ Face swapper model not found")
                return
            
            # Initialize with optimal providers for dynamic face swapping performance
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            try:
                self.face_swapper_model = insightface.model_zoo.get_model(
                    model_path, providers=providers
                )
                print("✅ Dynamic face swapper initialized with InsightFace")
                print("💡 Using proper neural network for expression-aware face swapping")
            except Exception as e:
                print(f"⚠️ Face swapper initialization error: {e}")
                # Fallback to CPU only
                try:
                    self.face_swapper_model = insightface.model_zoo.get_model(
                        model_path, providers=['CPUExecutionProvider']
                    )
                    print("✅ Dynamic face swapper initialized with CPU provider")
                except Exception as e2:
                    print(f"❌ Face swapper initialization failed: {e2}")
                    
        except ImportError:
            print("❌ InsightFace not available for fast face swapper")
        except Exception as e:
            print(f"❌ Face swapper initialization error: {e}")
            
    def set_source_face(self, source_image: np.ndarray, source_face: Face):
        """Set the source face for swapping with template extraction"""
        try:
            # Store the source face for neural network fallback
            self.source_face = source_face
            
            # Extract face using bbox (landmarks might not be reliable)
            bbox = source_face.bbox.astype(np.int32)
            x1, y1, x2, y2 = bbox
            
            # Ensure bbox is within image bounds with padding
            h, w = source_image.shape[:2]
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            if x2 > x1 and y2 > y1:
                # Extract face region
                face_region = source_image[y1:y2, x1:x2].copy()
                
                # Store multiple sizes for different target sizes
                self.source_face_templates = {
                    128: cv2.resize(face_region, (128, 128)),
                    96: cv2.resize(face_region, (96, 96)),
                    64: cv2.resize(face_region, (64, 64))
                }
                
                print("✅ Source face set with template extraction")
                return True
            else:
                print("⚠️ Invalid bbox dimensions")
                
        except Exception as e:
            print(f"Error setting source face: {e}")
                
        return False
    
    def get_target_face_cached(self, frame: np.ndarray) -> Optional[Face]:
        """Get target face with adaptive caching for smooth tracking"""
        current_time = time.time()
        
        # Always try to detect face for smooth tracking
        # Only use caching as a fallback for performance
        try:
            from modules.face_analyser import get_one_face
            target_face = get_one_face(frame)
            if target_face:
                self.cached_target_face = target_face
                self.face_cache_time = current_time
                return target_face
        except Exception as e:
            print(f"Face detection error: {e}")
        
        # Return cached face if detection failed and cache is recent
        if (self.cached_target_face is not None and 
            current_time - self.face_cache_time < 0.5):  # Use cache for max 500ms
            return self.cached_target_face
        
        return None
    
    def swap_face(self, frame: np.ndarray, target_face: Face) -> np.ndarray:
        """Dynamic face swap using proper InsightFace model with expression preservation"""
        if not self.face_swapper_model or not hasattr(self, 'source_face'):
            return frame
            
        start_time = time.time()
        
        try:
            # Use InsightFace proper face swapping for dynamic expressions
            # This preserves target expressions while applying source identity
            result = self.face_swapper_model.get(
                frame, target_face, self.source_face, paste_back=True
            )
            
            # Performance tracking
            end_time = time.time()
            frame_time = end_time - start_time
            self.frame_count += 1
            self.total_time += frame_time
            
            if self.frame_count % 300 == 0:  # Print stats every 300 frames to reduce overhead
                avg_time = self.total_time / self.frame_count * 1000
                fps = 1000 / avg_time if avg_time > 0 else 0
                print(f"Dynamic face swap: {avg_time:.1f}ms avg, {fps:.1f} FPS")
            
            return result
                    
        except Exception as e:
            # Only print error occasionally to avoid spam
            if self.frame_count % 60 == 0:
                print(f"Face swap error: {e}")
        
        return frame
    
    def _ultra_fast_swap(self, frame: np.ndarray, target_bbox: np.ndarray, target_kps: Optional[np.ndarray], start_time: float) -> np.ndarray:
        """Ultra-fast template-based face swapping with landmark alignment"""
        x1, y1, x2, y2 = target_bbox
        
        # Add padding for better blending
        padding = 15
        h, w = frame.shape[:2]
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)
        
        face_width = x2_pad - x1_pad
        face_height = y2_pad - y1_pad
        
        if face_width < 32 or face_height < 32:
            return frame
        
        # Choose appropriate template size based on target size
        if face_width >= 120:
            template_size = 128
        elif face_width >= 80:
            template_size = 96
        else:
            template_size = 64
            
        source_template = self.source_face_templates.get(template_size, self.source_face_templates[128])
        
        # Apply simple alignment based on face center for better positioning
        resized_source = cv2.resize(source_template, (face_width, face_height))
        
        # If we have keypoints, apply basic alignment
        if target_kps is not None and len(target_kps) >= 5:
            try:
                resized_source = self._align_face_simple(resized_source, target_kps, x1_pad, y1_pad, face_width, face_height)
            except Exception as e:
                # Fall back to simple resize if alignment fails
                pass
        
        # Create result frame
        result = frame.copy()
        target_region = result[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Create optimized blend mask with better eye region handling
        mask = self._create_fast_mask(face_height, face_width, target_kps, x1_pad, y1_pad)
        
        # Apply color adaptation and dynamic expression blending
        adapted_source = self._fast_color_adaptation(resized_source, target_region)
        
        # Add subtle target face features for expression preservation
        adapted_source = self._blend_expressions(adapted_source, target_region, face_width, face_height)
        
        # Ultra-fast blending
        target_float = target_region.astype(np.float32)
        source_float = adapted_source.astype(np.float32)
        
        # Very aggressive blending for solid face replacement
        alpha = 0.92  # Much higher for more solid face swapping
        
        # Create a more sophisticated blend - preserve some target features for expressions
        # Blend differently for different face regions
        eye_region_y = int(face_height * 0.25)
        mouth_region_y = int(face_height * 0.65)
        
        # Create region-aware blending
        blended = target_float.copy()
        
        # Upper face (forehead, eyes) - moderate blending to preserve expressions
        if eye_region_y > 0:
            upper_alpha = alpha * 0.8  # Slightly less aggressive for eye expressions
            upper_mask = mask[:eye_region_y]
            blended[:eye_region_y] = (target_float[:eye_region_y] * (1.0 - upper_mask * upper_alpha) + 
                                    source_float[:eye_region_y] * (upper_mask * upper_alpha))
        
        # Middle face (nose, cheeks) - full replacement
        middle_alpha = alpha
        middle_mask = mask[eye_region_y:mouth_region_y]
        blended[eye_region_y:mouth_region_y] = (target_float[eye_region_y:mouth_region_y] * (1.0 - middle_mask * middle_alpha) + 
                                              source_float[eye_region_y:mouth_region_y] * (middle_mask * middle_alpha))
        
        # Lower face (mouth, chin) - blend to preserve mouth movements
        if mouth_region_y < face_height:
            lower_alpha = alpha * 0.75  # Less aggressive for mouth expressions
            lower_mask = mask[mouth_region_y:]
            blended[mouth_region_y:] = (target_float[mouth_region_y:] * (1.0 - lower_mask * lower_alpha) + 
                                      source_float[mouth_region_y:] * (lower_mask * lower_alpha))
        
        result[y1_pad:y2_pad, x1_pad:x2_pad] = blended.astype(np.uint8)
        
        # Performance tracking
        end_time = time.time()
        frame_time = end_time - start_time
        self.frame_count += 1
        self.total_time += frame_time
        
        if self.frame_count % 300 == 0:  # Print stats every 300 frames to reduce overhead
            avg_time = self.total_time / self.frame_count * 1000
            fps = 1000 / avg_time if avg_time > 0 else 0
            print(f"Ultra-fast swap: {avg_time:.2f}ms avg, {fps:.1f} FPS")
        
        return result
    
    def _create_fast_mask(self, height: int, width: int, target_kps: Optional[np.ndarray] = None, x_offset: int = 0, y_offset: int = 0) -> np.ndarray:
        """Create optimized feathered mask with landmark-aware eye region handling"""
        # Create elliptical mask for natural face shape
        center_x, center_y = width // 2, height // 2
        
        # Adjust center if we have facial landmarks
        if target_kps is not None and len(target_kps) >= 5:
            try:
                # Adjust keypoints to cropped region
                adj_kps = target_kps - [x_offset, y_offset]
                
                # Use nose position as better center reference
                nose_pos = adj_kps[2]  # Nose is typically index 2 in 5-point landmarks
                
                # Validate nose position is within bounds
                if 0 <= nose_pos[0] <= width and 0 <= nose_pos[1] <= height:
                    # Shift center slightly toward nose for better alignment
                    center_x = int(center_x * 0.7 + nose_pos[0] * 0.3)
                    center_y = int(center_y * 0.8 + nose_pos[1] * 0.2)  # Less vertical shift
            except Exception:
                pass  # Fall back to geometric center
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        # Elliptical mask parameters - balanced for solid swapping with good coverage
        a = width * 0.44   # Larger for more complete face coverage
        b = height * 0.40  # Larger for more complete face coverage
        
        # Create elliptical mask with gradient falloff
        ellipse_mask = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2
        ellipse_mask = np.clip(1.0 - ellipse_mask, 0, 1)
        
        # Create solid center with smooth edges for better integration
        ellipse_mask = cv2.GaussianBlur(ellipse_mask.astype(np.float32), (15, 15), 5)
        
        # Apply power function for stronger center, softer edges
        ellipse_mask = ellipse_mask ** 0.7  # More aggressive for solid face replacement
        
        # Normalize and expand to 3 channels
        ellipse_mask = ellipse_mask / ellipse_mask.max() if ellipse_mask.max() > 0 else ellipse_mask
        
        return ellipse_mask[:, :, np.newaxis]
    
    def _fast_color_adaptation(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Fast color adaptation between source and target"""
        # Enhanced color matching for more natural results
        source_float = source.astype(np.float32)
        target_float = target.astype(np.float32)
        
        # Calculate means and standard deviations for better color matching
        source_mean = np.mean(source_float, axis=(0, 1))
        target_mean = np.mean(target_float, axis=(0, 1))
        
        # Enhanced color correction for solid, realistic face replacement
        color_diff = target_mean - source_mean
        adapted = source_float + color_diff * 0.45  # Stronger color matching for better integration
        
        # Enhanced brightness and contrast adjustment
        source_brightness = np.mean(source_mean)
        target_brightness = np.mean(target_mean)
        brightness_diff = target_brightness - source_brightness
        adapted += brightness_diff * 0.35  # Stronger brightness matching
        
        # Add slight contrast enhancement for more solid appearance
        contrast_factor = 1.05  # Subtle contrast boost
        adapted = (adapted - 127.5) * contrast_factor + 127.5
        
        return np.clip(adapted, 0, 255).astype(np.uint8)
    
    def _blend_expressions(self, source: np.ndarray, target: np.ndarray, width: int, height: int) -> np.ndarray:
        """Blend target expressions into source face for more dynamic appearance"""
        try:
            # Convert to float for blending
            source_float = source.astype(np.float32)
            target_float = target.astype(np.float32)
            
            # Define expression regions
            eye_region_start = int(height * 0.2)
            eye_region_end = int(height * 0.5)
            mouth_region_start = int(height * 0.6)
            mouth_region_end = int(height * 0.85)
            
            # Blend eye region for eye expressions (blinking, etc.)
            if eye_region_end > eye_region_start:
                eye_blend_factor = 0.15  # Subtle blend for eye expressions
                source_float[eye_region_start:eye_region_end] = (
                    source_float[eye_region_start:eye_region_end] * (1 - eye_blend_factor) +
                    target_float[eye_region_start:eye_region_end] * eye_blend_factor
                )
            
            # Blend mouth region for mouth expressions
            if mouth_region_end > mouth_region_start:
                mouth_blend_factor = 0.2  # More blend for mouth expressions
                source_float[mouth_region_start:mouth_region_end] = (
                    source_float[mouth_region_start:mouth_region_end] * (1 - mouth_blend_factor) +
                    target_float[mouth_region_start:mouth_region_end] * mouth_blend_factor
                )
            
            return np.clip(source_float, 0, 255).astype(np.uint8)
            
        except Exception as e:
            return source  # Return original on error
    
    def _align_face_simple(self, source: np.ndarray, target_kps: np.ndarray, x_offset: int, y_offset: int, width: int, height: int) -> np.ndarray:
        """Simple face alignment using eye positions to reduce double-vision effect"""
        try:
            # Get eye positions from 5-point landmarks (assuming standard order: left_eye, right_eye, nose, left_mouth, right_mouth)
            if len(target_kps) >= 2:
                # Adjust keypoints to the cropped region coordinates
                adj_kps = target_kps - [x_offset, y_offset]
                
                left_eye = adj_kps[0]   # Left eye
                right_eye = adj_kps[1]  # Right eye
                
                # Calculate the angle between eyes
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Only apply rotation if angle is reasonable (avoid extreme rotations)
                if abs(angle) > 5 and abs(angle) < 45:  # Only rotate for moderate head tilts
                    # Get rotation center (middle of the face)
                    center = (width // 2, height // 2)
                    
                    # Create rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
                    
                    # Apply rotation
                    aligned_source = cv2.warpAffine(
                        source, rotation_matrix, (width, height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT
                    )
                    
                    return aligned_source
                    
            return source  # Return original if alignment not needed/possible
            
        except Exception as e:
            return source  # Fallback to original on any error

# Global instance
_fast_swapper = None
_swapper_lock = threading.Lock()

def get_fast_swapper():
    """Get or create fast face swapper instance"""
    global _fast_swapper
    
    with _swapper_lock:
        if _fast_swapper is None:
            _fast_swapper = FastFaceSwapper()
        return _fast_swapper

def process_frame_fast(source_face: Face, frame: np.ndarray, source_image: Optional[np.ndarray] = None) -> np.ndarray:
    """Process frame with fast face swapping using proper alignment"""
    swapper = get_fast_swapper()
    
    # Set source face if provided
    if source_face is not None and not hasattr(swapper, 'source_face'):
        swapper.set_source_face(source_image, source_face)
    
    # Get target face with caching for smooth tracking
    target_face = swapper.get_target_face_cached(frame)
    if target_face:
        return swapper.swap_face(frame, target_face)
    
    return frame