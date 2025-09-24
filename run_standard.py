#!/usr/bin/env python3
"""
Standard Deep Live Cam launcher with fallback for missing dependencies
"""

import os
import sys
import platform

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Check if running on Apple Silicon
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.processor() == 'arm'

def check_dependencies():
    """Check if all dependencies are available"""
    missing_deps = []

    try:
        import cv2
    except ImportError:
        missing_deps.append('opencv-python')

    try:
        import insightface
    except ImportError:
        missing_deps.append('insightface')

    try:
        import torch
    except ImportError:
        missing_deps.append('torch')

    try:
        import onnxruntime
    except ImportError:
        missing_deps.append('onnxruntime')

    if missing_deps:
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install dependencies:")
        print("pip install -r requirements.txt")
        return False

    return True

def main():
    print("Deep Live Cam - Mac M3 Performance Optimized Edition")
    print("=" * 55)

    if IS_APPLE_SILICON:
        print("🍎 Apple Silicon detected - optimizations available")
    else:
        print("💻 Standard CPU mode")

    # Check dependencies first
    if not check_dependencies():
        return False

    # Try to import the optimized core
    try:
        from modules.core_optimized import run
        print("✓ Using optimized core with M3 enhancements")
        run()
        return True
    except ImportError as e:
        print(f"⚠️  Optimized core not available: {e}")
        print("Falling back to standard core...")

        # Fallback to standard core
        try:
            from modules.core import run
            print("✓ Using standard core")
            run()
            return True
        except ImportError as e:
            print(f"✗ Standard core not available: {e}")
            return False
    except Exception as e:
        print(f"✗ Error running application: {e}")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)