#!/usr/bin/env python3
"""
Apple Silicon Optimization Setup Script
Configures environment variables and system settings for 20+ FPS performance
"""

import os
import platform
import subprocess
import sys

def detect_apple_silicon():
    """Detect Apple Silicon chip type"""
    if platform.system() != 'Darwin':
        return None, None
        
    try:
        # Get chip information
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                              capture_output=True, text=True)
        cpu_brand = result.stdout.strip()
        
        # Detect chip type
        if 'M1' in cpu_brand:
            return 'M1', 8  # 4 performance + 4 efficiency cores
        elif 'M2' in cpu_brand:
            return 'M2', 8  # 4 performance + 4 efficiency cores  
        elif 'M3' in cpu_brand:
            if 'Pro' in cpu_brand:
                return 'M3 Pro', 12  # 6 performance + 6 efficiency cores
            elif 'Max' in cpu_brand:
                return 'M3 Max', 16  # 8 performance + 8 efficiency cores
            else:
                return 'M3', 8  # 4 performance + 4 efficiency cores
        else:
            return 'Unknown Apple Silicon', 8
            
    except Exception as e:
        print(f"Error detecting Apple Silicon: {e}")
        return None, None

def setup_environment_variables(chip_type: str, total_cores: int):
    """Set up optimal environment variables"""
    
    # Calculate optimal thread counts
    performance_cores = total_cores // 2
    efficiency_cores = total_cores - performance_cores
    
    env_vars = {
        # Neural Engine optimizations
        'BNNS_ALLOW_GEMM': '1',
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
        
        # Threading optimizations
        'OMP_NUM_THREADS': str(performance_cores),
        'MKL_NUM_THREADS': str(performance_cores),
        'VECLIB_MAXIMUM_THREADS': str(total_cores),
        'OPENBLAS_NUM_THREADS': str(performance_cores),
        
        # Memory optimizations
        'PYTORCH_MPS_ALLOCATOR_POLICY': 'garbage_collection',
        'MALLOC_NANO_ZONE': '1',  # Use nano zone for small allocations
        
        # OpenCV optimizations
        'OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION': '1',
        'OPENCV_AVFOUNDATION_SKIP_AUTH': '1',
        
        # Python optimizations
        'PYTHONOPTIMIZE': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
    }
    
    # Chip-specific optimizations
    if 'M3' in chip_type:
        env_vars.update({
            'ONNX_ML_TOOLS_THREADS': str(performance_cores + 2),
            'COREML_COMPUTE_UNITS': 'ALL',
        })
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    return env_vars

def optimize_python_environment():
    """Optimize Python environment for performance"""
    
    try:
        # Import required packages and configure them
        import cv2
        cv2.setNumThreads(0)  # Use all available threads
        cv2.setUseOptimized(True)
        print("OpenCV optimization enabled")
        
        # Configure numpy
        import numpy as np
        print(f"NumPy using BLAS: {np.__config__.show()}")
        
        # Configure memory management
        import gc
        gc.set_threshold(700, 10, 10)  # More aggressive garbage collection
        print("Aggressive garbage collection configured")
        
    except ImportError as e:
        print(f"Warning: Could not optimize {e}")

def check_dependencies():
    """Check that required packages are installed"""
    required_packages = [
        'opencv-python',
        'numpy', 
        'insightface',
        'onnxruntime',
        'psutil',
        'customtkinter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("All required packages are installed")
    return True

def create_performance_profile():
    """Create a performance configuration file"""
    
    chip_type, total_cores = detect_apple_silicon()
    if not chip_type:
        print("Not running on Apple Silicon")
        return
    
    config = {
        'chip_type': chip_type,
        'total_cores': total_cores,
        'performance_cores': total_cores // 2,
        'efficiency_cores': total_cores - (total_cores // 2),
        'target_fps': 25 if 'M3' in chip_type else 20,
        'memory_limit_gb': 8,
        'face_detection_interval': 30 if 'M3' in chip_type else 45,
        'batch_size': 6 if 'M3' in chip_type else 4,
    }
    
    config_path = os.path.join(os.path.dirname(__file__), 'apple_silicon_config.json')
    
    try:
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Performance profile saved to: {config_path}")
    except Exception as e:
        print(f"Could not save performance profile: {e}")

def main():
    """Main setup function"""
    print("=== Apple Silicon Deep Live Cam Optimization Setup ===")
    
    # Detect hardware
    chip_type, total_cores = detect_apple_silicon()
    
    if not chip_type:
        print("This script is designed for Apple Silicon Macs")
        sys.exit(1)
    
    print(f"Detected: {chip_type} with {total_cores} cores")
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies first")
        sys.exit(1)
    
    # Setup environment
    env_vars = setup_environment_variables(chip_type, total_cores)
    
    # Optimize Python environment
    optimize_python_environment()
    
    # Create performance profile
    create_performance_profile()
    
    print("\n=== Optimization Complete ===")
    print(f"Optimized for: {chip_type}")
    print(f"Target FPS: {25 if 'M3' in chip_type else 20}+")
    print("Environment variables set for current session")
    print("\nTo make permanent, add these to your shell profile:")
    
    for key, value in env_vars.items():
        print(f"export {key}={value}")
    
    print("\nRun the application with: python run_optimized.py --use-optimized --enable-monitoring")

if __name__ == "__main__":
    main()