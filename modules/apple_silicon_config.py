"""
Apple Silicon M3 specific configuration and optimizations
"""

import os
import platform
import subprocess
import threading
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass

# Check if we're on Apple Silicon
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.processor() == 'arm'


@dataclass
class AppleSiliconSpecs:
    """Apple Silicon hardware specifications"""
    model: str = "Unknown"
    performance_cores: int = 8
    efficiency_cores: int = 4
    gpu_cores: int = 10
    neural_engine_tops: float = 18.0
    unified_memory_gb: int = 8
    memory_bandwidth_gbps: int = 100


def detect_apple_silicon_specs() -> AppleSiliconSpecs:
    """Detect Apple Silicon hardware specifications"""
    specs = AppleSiliconSpecs()

    if not IS_APPLE_SILICON:
        return specs

    try:
        # Get system information
        result = subprocess.run(['system_profiler', 'SPHardwareDataType', '-json'],
                               capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            hardware = data.get('SPHardwareDataType', [{}])[0]

            # Extract model information
            chip_type = hardware.get('chip_type', '')
            if 'M3' in chip_type:
                if 'Pro' in chip_type:
                    specs.model = "M3 Pro"
                    specs.performance_cores = 12
                    specs.efficiency_cores = 4
                    specs.gpu_cores = 18
                    specs.neural_engine_tops = 35.0
                    specs.unified_memory_gb = 18
                    specs.memory_bandwidth_gbps = 150
                elif 'Max' in chip_type:
                    specs.model = "M3 Max"
                    specs.performance_cores = 12
                    specs.efficiency_cores = 4
                    specs.gpu_cores = 30
                    specs.neural_engine_tops = 35.0
                    specs.unified_memory_gb = 36
                    specs.memory_bandwidth_gbps = 300
                else:
                    specs.model = "M3"
                    specs.performance_cores = 8
                    specs.efficiency_cores = 4
                    specs.gpu_cores = 10
                    specs.neural_engine_tops = 18.0
                    specs.unified_memory_gb = 8
                    specs.memory_bandwidth_gbps = 100

            # Get actual memory size
            memory_str = hardware.get('physical_memory', '8 GB')
            memory_gb = int(memory_str.split()[0]) if memory_str else 8
            specs.unified_memory_gb = memory_gb

    except Exception as e:
        print(f"Warning: Could not detect Apple Silicon specs: {e}")

    return specs


class AppleSiliconOptimizer:
    """Apple Silicon specific optimizations"""

    def __init__(self):
        self.specs = detect_apple_silicon_specs()
        self.optimization_level = "balanced"  # performance, balanced, efficiency
        self._setup_environment()

    def _setup_environment(self):
        """Setup optimal environment variables"""
        if not IS_APPLE_SILICON:
            return

        # Neural Engine optimizations
        os.environ['BNNS_ALLOW_GEMM'] = '1'
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

        # Metal optimizations
        os.environ['MTL_HUD_ENABLED'] = '0'  # Disable HUD for performance
        os.environ['MTL_DEBUG_LAYER'] = '0'  # Disable debug layer

        # Core optimizations
        total_cores = self.specs.performance_cores + self.specs.efficiency_cores
        os.environ['OMP_NUM_THREADS'] = str(self.specs.performance_cores)
        os.environ['MKL_NUM_THREADS'] = str(self.specs.performance_cores)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(total_cores)

        # Memory optimizations
        memory_gb = min(self.specs.unified_memory_gb, 16)  # Cap at 16GB for safety
        os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'

    def get_optimal_execution_providers(self) -> list:
        """Get optimal execution providers for the detected hardware"""
        if not IS_APPLE_SILICON:
            return ['CPUExecutionProvider']

        providers = []

        # Always prefer CoreML for Apple Silicon
        providers.append('CoreMLExecutionProvider')

        # Add CPU as fallback
        providers.append('CPUExecutionProvider')

        return providers

    def get_session_options(self) -> dict:
        """Get optimal ONNX session options"""
        options = {
            'graph_optimization_level': 'ORT_ENABLE_ALL',
            'inter_op_num_threads': self.specs.performance_cores,
            'intra_op_num_threads': self.specs.performance_cores,
            'enable_mem_pattern': True,
            'enable_mem_reuse': True,
            'execution_mode': 'ORT_PARALLEL'
        }

        # Adjust based on optimization level
        if self.optimization_level == "performance":
            options['inter_op_num_threads'] = self.specs.performance_cores + 2
            options['intra_op_num_threads'] = self.specs.performance_cores + 2
        elif self.optimization_level == "efficiency":
            options['inter_op_num_threads'] = max(2, self.specs.performance_cores // 2)
            options['intra_op_num_threads'] = max(2, self.specs.performance_cores // 2)

        return options

    def get_opencv_optimizations(self) -> dict:
        """Get OpenCV optimization settings"""
        return {
            'threads': self.specs.performance_cores,
            'use_optimized': True,
            'use_opencl': False,  # Not needed on Apple Silicon
            'backend_preference': 'DNN_BACKEND_DEFAULT'
        }

    def get_batch_size_recommendation(self) -> int:
        """Recommend batch size based on hardware with M1 BASE optimization"""
        # M1 BASE-specific: Lower batch sizes due to lower compute and memory bandwidth
        # Exclude M1 Pro/Max/Ultra which should use standard settings
        is_m1_base = ("M1" in self.specs.model and "M2" not in self.specs.model and "M3" not in self.specs.model
                      and "Pro" not in self.specs.model and "Max" not in self.specs.model and "Ultra" not in self.specs.model)

        if is_m1_base:
            # M1 optimized batch sizes (more conservative)
            if self.specs.unified_memory_gb >= 16:
                return 3  # Reduced from 6
            elif self.specs.unified_memory_gb >= 8:
                return 2  # Reduced from 4 (M1 base model)
            else:
                return 1
        else:
            # M2/M3 batch sizes (original)
            if self.specs.unified_memory_gb >= 32:
                return 8
            elif self.specs.unified_memory_gb >= 16:
                return 6
            elif self.specs.unified_memory_gb >= 8:
                return 4
            else:
                return 2

    def get_frame_buffer_size(self) -> int:
        """Recommend frame buffer size with M1 BASE optimization"""
        # M1 BASE-specific: Smaller buffer pools for better cache locality
        # Exclude M1 Pro/Max/Ultra which should use standard settings
        is_m1_base = ("M1" in self.specs.model and "M2" not in self.specs.model and "M3" not in self.specs.model
                      and "Pro" not in self.specs.model and "Max" not in self.specs.model and "Ultra" not in self.specs.model)

        if is_m1_base:
            # M1 optimized buffer sizes (smaller for cache efficiency)
            if self.specs.unified_memory_gb >= 16:
                return 8  # Reduced from 15
            elif self.specs.unified_memory_gb >= 8:
                return 5  # Reduced from 10 (M1 base model)
            else:
                return 3
        else:
            # M2/M3 buffer sizes (original)
            if self.specs.unified_memory_gb >= 32:
                return 20
            elif self.specs.unified_memory_gb >= 16:
                return 15
            elif self.specs.unified_memory_gb >= 8:
                return 10
            else:
                return 5

    def get_quality_settings(self) -> dict:
        """Get recommended quality settings with M1 BASE-specific optimization"""

        # OPTIMIZATION 1.2: M1 BASE-specific resolution tuning
        # Detect M1 BASE vs M2/M3/M1 Pro/Max/Ultra
        # Exclude M1 Pro/Max/Ultra which should use standard settings
        is_m1_base = ("M1" in self.specs.model and "M2" not in self.specs.model and "M3" not in self.specs.model
                      and "Pro" not in self.specs.model and "Max" not in self.specs.model and "Ultra" not in self.specs.model)

        if is_m1_base:
            # M1-optimized settings: Lower resolution for 20 FPS target
            base_settings = {
                "performance": {
                    "resolution": (640, 480),  # 640×480 (307K pixels, -41% vs 960×540)
                    "fps_target": 25,          # Realistic target for M1
                    "quality": 25,
                    "encoder": "hevc_videotoolbox",
                    "description": "Maximum FPS mode for M1"
                },
                "balanced": {
                    "resolution": (720, 540),  # 720×540 (389K pixels, -25% vs 960×540)
                    "fps_target": 20,          # 20 FPS TARGET for M1
                    "quality": 23,
                    "encoder": "hevc_videotoolbox",
                    "description": "Recommended for M1 - 20 FPS target"
                },
                "quality": {
                    "resolution": (960, 720),  # 960×720 (691K pixels)
                    "fps_target": 15,          # Lower target for quality
                    "quality": 20,
                    "encoder": "hevc_videotoolbox",
                    "description": "Best quality mode for M1"
                }
            }
            print(f"[AppleSiliconConfig] M1 detected: Using optimized resolution presets")
            print(f"  - Performance: 640×480 @ 25 FPS target")
            print(f"  - Balanced:    720×540 @ 20 FPS target (RECOMMENDED)")
            print(f"  - Quality:     960×720 @ 15 FPS target")
        else:
            # M2/M3 settings: Keep original higher resolutions
            base_settings = {
                "performance": {
                    "resolution": (640, 480),
                    "fps_target": 60,
                    "quality": 23,
                    "encoder": "hevc_videotoolbox",
                    "description": "Maximum FPS mode"
                },
                "balanced": {
                    "resolution": (960, 540),
                    "fps_target": 30,
                    "quality": 20,
                    "encoder": "hevc_videotoolbox",
                    "description": "Recommended general use"
                },
                "quality": {
                    "resolution": (1280, 720),
                    "fps_target": 24,
                    "quality": 18,
                    "encoder": "hevc_videotoolbox",
                    "description": "Best quality mode"
                }
            }

            # Adjust for M3 Pro/Max
            if "Pro" in self.specs.model or "Max" in self.specs.model:
                base_settings["performance"]["fps_target"] = 120
                base_settings["balanced"]["fps_target"] = 60
                base_settings["quality"]["resolution"] = (1920, 1080)

        return base_settings

    def enable_thermal_monitoring(self) -> bool:
        """Enable thermal monitoring for Apple Silicon"""
        try:
            # Check if thermal monitoring tools are available
            result = subprocess.run(['which', 'powermetrics'],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def get_thermal_state(self) -> str:
        """Get current thermal state"""
        if not IS_APPLE_SILICON:
            return "unknown"

        try:
            result = subprocess.run(['pmset', '-g', 'thermlog'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                output = result.stdout.lower()
                if 'nominal' in output:
                    return "nominal"
                elif 'fair' in output:
                    return "fair"
                elif 'serious' in output:
                    return "serious"
                elif 'critical' in output:
                    return "critical"
        except:
            pass

        return "unknown"

    def adjust_for_thermal_state(self, thermal_state: str) -> dict:
        """Adjust settings based on thermal state"""
        adjustments = {
            "nominal": {"scale_factor": 1.0, "batch_size_scale": 1.0},
            "fair": {"scale_factor": 0.9, "batch_size_scale": 0.8},
            "serious": {"scale_factor": 0.7, "batch_size_scale": 0.6},
            "critical": {"scale_factor": 0.5, "batch_size_scale": 0.4},
            "unknown": {"scale_factor": 0.8, "batch_size_scale": 0.8}
        }

        return adjustments.get(thermal_state, adjustments["unknown"])

    def print_system_info(self):
        """Print detected system information"""
        print(f"Apple Silicon Model: {self.specs.model}")
        print(f"Performance Cores: {self.specs.performance_cores}")
        print(f"Efficiency Cores: {self.specs.efficiency_cores}")
        print(f"GPU Cores: {self.specs.gpu_cores}")
        print(f"Neural Engine: {self.specs.neural_engine_tops} TOPS")
        print(f"Unified Memory: {self.specs.unified_memory_gb} GB")
        print(f"Memory Bandwidth: {self.specs.memory_bandwidth_gbps} GB/s")
        print(f"Optimization Level: {self.optimization_level}")


# Global instance
_optimizer_instance: Optional[AppleSiliconOptimizer] = None
_optimizer_lock = threading.Lock()


def get_apple_silicon_optimizer() -> AppleSiliconOptimizer:
    """Get singleton Apple Silicon optimizer"""
    global _optimizer_instance

    with _optimizer_lock:
        if _optimizer_instance is None:
            _optimizer_instance = AppleSiliconOptimizer()

    return _optimizer_instance


def configure_for_apple_silicon():
    """Configure the entire application for Apple Silicon"""
    if not IS_APPLE_SILICON:
        print("Not running on Apple Silicon - skipping optimizations")
        return

    optimizer = get_apple_silicon_optimizer()
    optimizer.print_system_info()

    # Apply all optimizations
    print("Applying Apple Silicon optimizations...")
    return optimizer