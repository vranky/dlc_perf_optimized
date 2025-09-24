#!/usr/bin/env python3
"""
Quick test script to verify Mac M3 optimizations are working
"""

import sys
import os
import platform
import time

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def test_imports():
    """Test that all optimized modules can be imported"""
    print("Testing module imports...")

    try:
        from modules.performance_optimizer import FPSMonitor, FrameBufferPool
        print("✓ performance_optimizer imported successfully")
    except Exception as e:
        print(f"✗ performance_optimizer failed: {e}")
        return False

    try:
        from modules.apple_silicon_config import get_apple_silicon_optimizer, IS_APPLE_SILICON
        print(f"✓ apple_silicon_config imported successfully (Apple Silicon: {IS_APPLE_SILICON})")
    except Exception as e:
        print(f"✗ apple_silicon_config failed: {e}")
        return False

    try:
        from modules.processors.frame.face_swapper_optimized import get_optimized_face_swapper
        print("✓ face_swapper_optimized imported successfully")
    except Exception as e:
        print(f"✗ face_swapper_optimized failed: {e}")
        return False

    return True

def test_apple_silicon_detection():
    """Test Apple Silicon detection"""
    print("\nTesting Apple Silicon detection...")

    from modules.apple_silicon_config import get_apple_silicon_optimizer, IS_APPLE_SILICON

    if IS_APPLE_SILICON:
        print("✓ Running on Apple Silicon")
        optimizer = get_apple_silicon_optimizer()
        optimizer.print_system_info()
    else:
        print("ℹ Not running on Apple Silicon - optimizations will use fallback mode")

    return True

def test_fps_monitoring():
    """Test FPS monitoring functionality"""
    print("\nTesting FPS monitoring...")

    from modules.performance_optimizer import FPSMonitor

    monitor = FPSMonitor(window_size=10)

    # Simulate some frame processing
    for i in range(20):
        monitor.start_frame()
        time.sleep(0.016)  # ~60 FPS simulation
        monitor.end_frame()

    metrics = monitor.get_metrics()
    print(f"✓ FPS Monitor working: {metrics.fps:.1f} FPS, {metrics.frame_time:.1f}ms")

    return True

def test_buffer_pool():
    """Test frame buffer pool"""
    print("\nTesting frame buffer pool...")

    from modules.performance_optimizer import FrameBufferPool

    pool = FrameBufferPool(pool_size=5, frame_shape=(480, 640, 3))

    # Get and return buffers
    buffer1 = pool.get_buffer()
    buffer2 = pool.get_buffer()

    print(f"✓ Buffer pool working: shape {buffer1.shape}, dtype {buffer1.dtype}")

    pool.return_buffer(buffer1)
    pool.return_buffer(buffer2)

    return True

def test_opencv_optimizations():
    """Test OpenCV optimizations"""
    print("\nTesting OpenCV optimizations...")

    try:
        import cv2
        import numpy as np
        from modules.performance_optimizer import optimize_opencv_settings

        # Apply optimizations
        optimize_opencv_settings()

        # Test basic operations
        test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        start_time = time.perf_counter()
        for _ in range(10):
            resized = cv2.resize(test_image, (320, 240))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        end_time = time.perf_counter()

        ops_per_sec = 10 / (end_time - start_time)
        print(f"✓ OpenCV operations: {ops_per_sec:.1f} ops/sec")

        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Mac M3 Optimization Test Suite")
    print("=" * 50)

    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()}")

    tests = [
        test_imports,
        test_apple_silicon_detection,
        test_fps_monitoring,
        test_buffer_pool,
        test_opencv_optimizations
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 50)
    print("Test Results")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Mac M3 optimizations are ready.")
        print("\nNext steps:")
        print("1. Run benchmark: python benchmark_performance.py")
        print("2. Test with camera: python run_optimized.py")
        print("3. Process video: python run_optimized.py -s face.jpg -t video.mp4 -o output.mp4")
        return True
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)