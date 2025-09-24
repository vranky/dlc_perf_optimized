#!/usr/bin/env python3
"""
Debug script to identify issues with the optimized version
"""

import os
import sys
import platform
import traceback

print("Deep Live Cam - Debug Information")
print("=" * 40)

# System info
print(f"Platform: {platform.platform()}")
print(f"Python: {platform.python_version()}")
print(f"Architecture: {platform.processor()}")
print(f"Current directory: {os.getcwd()}")

# Check if we're on Apple Silicon
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.processor() == 'arm'
print(f"Apple Silicon: {IS_APPLE_SILICON}")

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
print(f"Python path: {sys.path[:3]}...")

print("\nChecking dependencies...")

# Check basic imports
dependencies = [
    'numpy', 'cv2', 'torch', 'onnxruntime', 'insightface',
    'tkinter', 'customtkinter', 'PIL'
]

for dep in dependencies:
    try:
        if dep == 'cv2':
            import cv2
            print(f"✓ {dep} - version {cv2.__version__}")
        elif dep == 'torch':
            import torch
            print(f"✓ {dep} - version {torch.__version__}")
        elif dep == 'PIL':
            import PIL
            print(f"✓ {dep} - PIL available")
        elif dep == 'customtkinter':
            import customtkinter
            print(f"✓ {dep} - available")
        else:
            __import__(dep)
            print(f"✓ {dep} - available")
    except ImportError as e:
        print(f"✗ {dep} - {e}")
    except Exception as e:
        print(f"? {dep} - {e}")

print("\nChecking modules...")

# Check our modules
modules_to_check = [
    'modules.globals',
    'modules.core',
    'modules.processors.frame.core',
    'modules.processors.frame.face_swapper'
]

for module in modules_to_check:
    try:
        __import__(module)
        print(f"✓ {module}")
    except ImportError as e:
        print(f"✗ {module} - {e}")
    except Exception as e:
        print(f"? {module} - {e}")

print("\nChecking frame processors...")

# Check frame processor loading
try:
    from modules.processors.frame.core import load_frame_processor_module

    # Test standard face_swapper
    try:
        face_swapper = load_frame_processor_module('face_swapper')
        print("✓ face_swapper loaded successfully")

        # Check required methods
        required_methods = ['pre_check', 'pre_start', 'process_frame', 'process_image', 'process_video']
        for method in required_methods:
            if hasattr(face_swapper, method):
                print(f"  ✓ {method}")
            else:
                print(f"  ✗ {method} missing")

    except Exception as e:
        print(f"✗ face_swapper failed: {e}")
        traceback.print_exc()

    # Test optimized face_swapper
    try:
        face_swapper_opt = load_frame_processor_module('face_swapper_optimized')
        print("✓ face_swapper_optimized loaded successfully")

        # Check required methods
        required_methods = ['pre_check', 'pre_start', 'process_frame', 'process_image', 'process_video']
        for method in required_methods:
            if hasattr(face_swapper_opt, method):
                print(f"  ✓ {method}")
            else:
                print(f"  ✗ {method} missing")

    except Exception as e:
        print(f"✗ face_swapper_optimized failed: {e}")
        traceback.print_exc()

except Exception as e:
    print(f"✗ Frame processor core failed: {e}")
    traceback.print_exc()

print("\nChecking models directory...")
models_dir = os.path.join(os.getcwd(), 'models')
if os.path.exists(models_dir):
    print(f"✓ Models directory exists: {models_dir}")
    files = os.listdir(models_dir)
    for file in files:
        print(f"  - {file}")

    required_models = ['inswapper_128_fp16.onnx', 'GFPGANv1.4.pth']
    for model in required_models:
        if model in files:
            print(f"✓ {model} found")
        else:
            print(f"✗ {model} missing")
else:
    print(f"✗ Models directory not found: {models_dir}")

print("\nDebug completed!")
print("\nRecommended command to run:")
if IS_APPLE_SILICON:
    print("python run_m3_optimized.py --execution-provider coreml")
else:
    print("python run.py")