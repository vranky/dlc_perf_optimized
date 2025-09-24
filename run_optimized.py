#!/usr/bin/env python3
"""
Optimized Deep Live Cam launcher for Mac M3
Achieves >20 FPS performance on Apple Silicon
"""

import os
import sys
import platform
import argparse

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Configure environment for Apple Silicon
if platform.system() == 'Darwin' and platform.processor() == 'arm':
    # Apple Silicon optimizations
    from modules.apple_silicon_config import configure_for_apple_silicon
    optimizer = configure_for_apple_silicon()
    print("Apple Silicon M3 optimizations enabled")
else:
    print("Running on non-Apple Silicon hardware")

# Import optimized core
from modules.core_optimized import run

if __name__ == '__main__':
    print("Deep Live Cam - Optimized for Mac M3")
    print("Target: >20 FPS performance")
    print("-" * 40)

    # Add help for optimized arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("\nOptimized arguments:")
        print("--use-optimized       Use optimized processing (default: True)")
        print("--batch-size N        Processing batch size (default: 4)")
        print("--enable-monitoring   Enable FPS monitoring (default: True)")
        print("--video-encoder ENC   Video encoder (hevc_videotoolbox for Mac)")
        print("\nExample for live camera (optimized):")
        print("python run_optimized.py --use-optimized --enable-monitoring")
        print("\nExample for video processing:")
        print("python run_optimized.py -s face.jpg -t input.mp4 -o output.mp4 --use-optimized")

    # Run optimized core
    run()