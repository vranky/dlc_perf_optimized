#!/usr/bin/env python3
"""
Simple M3 optimized launcher that applies basic optimizations to standard core
"""

import os
import sys
import platform
import multiprocessing as mp

# Configure environment for Apple Silicon before imports
if platform.system() == 'Darwin' and platform.processor() == 'arm':
    # Apple Silicon optimizations
    os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['OPENBLAS_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(mp.cpu_count())
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['BNNS_ALLOW_GEMM'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    print("🚀 Applied Apple Silicon M3 environment optimizations")
else:
    # Single thread for CUDA
    if any(arg.startswith('--execution-provider') for arg in sys.argv):
        os.environ['OMP_NUM_THREADS'] = '1'

# Reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run standard core
if __name__ == '__main__':
    print("Deep Live Cam - M3 Optimized")
    print("Using standard core with M3 optimizations applied")
    print("-" * 50)

    # Modify arguments to use CoreML by default on Apple Silicon
    if platform.system() == 'Darwin' and '--execution-provider' not in ' '.join(sys.argv):
        sys.argv.extend(['--execution-provider', 'coreml'])
        print("Added CoreML execution provider for Apple Silicon")

    # Import and run
    from modules import core
    core.run()