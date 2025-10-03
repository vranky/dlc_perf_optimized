#!/usr/bin/env python3
"""
Comprehensive diagnostic script for Mac M1 performance issues
Identifies bottlenecks when FPS is severely degraded (e.g., 0.8 FPS instead of 24-28 FPS)
"""

import sys
import os
import time
import platform
import subprocess

# Add modules to path
sys.path.insert(0, os.path.dirname(__file__))

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def check_system_info():
    """Check system information"""
    print_section("SYSTEM INFORMATION")

    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Version: {platform.python_version()}")

    if platform.system() != 'Darwin':
        print("\n⚠️  WARNING: Not running on macOS!")
        print("   This optimization is specifically for Mac M1")
        return

    # Check if M1
    try:
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                              capture_output=True, text=True, timeout=1)
        cpu_brand = result.stdout.strip()
        print(f"CPU Brand: {cpu_brand}")

        if 'M1' in cpu_brand and 'M2' not in cpu_brand and 'M3' not in cpu_brand:
            print("✓ Mac M1 detected - Optimizations should be active")
        elif 'M2' in cpu_brand or 'M3' in cpu_brand:
            print(f"⚠️  {cpu_brand} detected - Using standard optimizations")
        else:
            print("⚠️  Not Apple Silicon - Performance will be degraded")

        # Check cores
        result = subprocess.run(['sysctl', '-n', 'hw.perflevel0.physicalcpu'],
                              capture_output=True, text=True, timeout=1)
        perf_cores = result.stdout.strip()
        print(f"Performance Cores: {perf_cores}")

        result = subprocess.run(['sysctl', '-n', 'hw.physicalcpu'],
                              capture_output=True, text=True, timeout=1)
        total_cores = result.stdout.strip()
        print(f"Total Physical Cores: {total_cores}")

        # Check memory
        result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                              capture_output=True, text=True, timeout=1)
        mem_bytes = int(result.stdout.strip())
        mem_gb = mem_bytes / (1024**3)
        print(f"Total Memory: {mem_gb:.1f} GB")

        # Check thermal state
        result = subprocess.run(['pmset', '-g', 'thermlog'],
                              capture_output=True, text=True, timeout=1)
        if result.returncode == 0:
            output = result.stdout.lower()
            if 'nominal' in output:
                print("✓ Thermal State: Nominal (Good)")
            elif 'fair' in output or 'moderate' in output:
                print("⚠️  Thermal State: Fair/Moderate (May throttle)")
            elif 'serious' in output or 'critical' in output:
                print("⚠️  Thermal State: Serious/Critical (THROTTLING)")

    except Exception as e:
        print(f"Could not get system info: {e}")


def check_python_packages():
    """Check critical Python packages"""
    print_section("PYTHON PACKAGES")

    critical_packages = {
        'numpy': None,
        'opencv-python': 'cv2',
        'onnxruntime': 'onnxruntime',
        'onnxruntime-silicon': None,  # Check separately
        'insightface': 'insightface',
        'torch': 'torch',
    }

    for package_name, import_name in critical_packages.items():
        try:
            if package_name == 'opencv-python':
                import cv2
                print(f"✓ opencv-python: {cv2.__version__}")
            elif package_name == 'onnxruntime':
                import onnxruntime
                print(f"✓ onnxruntime: {onnxruntime.__version__}")
            elif package_name == 'onnxruntime-silicon':
                # Check via pip
                import subprocess
                result = subprocess.run(['pip', 'show', 'onnxruntime-silicon'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    version_line = [l for l in result.stdout.split('\n') if 'Version:' in l]
                    if version_line:
                        version = version_line[0].split(':')[1].strip()
                        print(f"✓ onnxruntime-silicon: {version}")
                        if version != '1.16.3':
                            print(f"  ⚠️  Expected 1.16.3, got {version}")
                    else:
                        print(f"✓ onnxruntime-silicon: Installed (version unknown)")
                else:
                    print(f"✗ onnxruntime-silicon: NOT INSTALLED")
            elif package_name == 'insightface':
                import insightface
                print(f"✓ insightface: {insightface.__version__}")
            elif package_name == 'torch':
                import torch
                print(f"✓ torch: {torch.__version__}")
                # Check MPS availability
                if hasattr(torch.backends, 'mps'):
                    if torch.backends.mps.is_available():
                        print(f"  ✓ MPS (Metal) backend: Available")
                    else:
                        print(f"  ⚠️  MPS (Metal) backend: Not available")
            elif package_name == 'numpy':
                import numpy
                print(f"✓ numpy: {numpy.__version__}")
        except ImportError:
            print(f"✗ {package_name}: NOT INSTALLED")
        except Exception as e:
            print(f"✗ {package_name}: Error - {e}")


def check_onnx_providers():
    """Check ONNX Runtime execution providers"""
    print_section("ONNX RUNTIME PROVIDERS (CRITICAL)")

    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        print(f"Available providers: {available}")

        if 'CoreMLExecutionProvider' in available:
            print("\n✓ CoreML provider: AVAILABLE (GOOD for M1)")
            print("  This uses the Neural Engine (11 TOPS on M1)")
        else:
            print("\n✗ CoreML provider: NOT AVAILABLE (CRITICAL ISSUE)")
            print("  ⚠️  This is likely causing your 0.8 FPS performance!")
            print("  Without CoreML, everything runs on CPU (10-20x slower)")
            print("\nFix:")
            print("  pip uninstall onnxruntime onnxruntime-silicon -y")
            print("  pip install onnxruntime-silicon==1.16.3")
            print("  # Then restart Python and check again")

        if 'CPUExecutionProvider' in available:
            print("✓ CPU provider: Available (fallback)")

        # Additional provider info
        print(f"\nProvider priority order: {available}")
        print("Note: First provider in list is used if available")

    except Exception as e:
        print(f"✗ Error checking ONNX providers: {e}")
        import traceback
        traceback.print_exc()


def check_models():
    """Check if models exist and are accessible"""
    print_section("MODEL FILES")

    models_dir = os.path.join(os.path.dirname(__file__), "models")

    required_models = {
        "inswapper_128_fp16.onnx": (125, 155),  # Expected size range in MB
        "GFPGANv1.4.pth": (340, 360),
    }

    print(f"Models directory: {models_dir}")

    if not os.path.exists(models_dir):
        print(f"✗ Models directory does not exist!")
        print(f"  Create it: mkdir -p {models_dir}")
        return False

    all_found = True
    for model, (min_mb, max_mb) in required_models.items():
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if min_mb <= size_mb <= max_mb:
                print(f"✓ {model}: Found ({size_mb:.1f} MB) - Size OK")
            else:
                print(f"⚠️  {model}: Found ({size_mb:.1f} MB) - Expected {min_mb}-{max_mb} MB")
                print(f"   File may be corrupted or incorrect version")
                all_found = False
        else:
            print(f"✗ {model}: NOT FOUND (CRITICAL)")
            print(f"   Download and place in {models_dir}/")
            all_found = False

    return all_found


def test_face_detection_speed():
    """Test face detection speed"""
    print_section("FACE DETECTION PERFORMANCE TEST")

    try:
        import cv2
        import numpy as np
        import insightface
        from insightface.app import FaceAnalysis
        import onnxruntime as ort

        print("Creating face analyzer...")

        # Get execution providers
        providers = ort.get_available_providers()
        print(f"Using providers: {providers[:2]}")

        # Create analyzer
        start = time.time()
        app = FaceAnalysis(name='buffalo_l', providers=providers[:2])
        app.prepare(ctx_id=0, det_size=(640, 640))
        init_time = time.time() - start
        print(f"✓ Face analyzer initialized in {init_time:.2f}s")

        # Test on dummy image (720×540 - M1 optimized resolution)
        print("\nTesting face detection on 720×540 image...")
        test_img = np.random.randint(0, 255, (540, 720, 3), dtype=np.uint8)

        # Warm-up (first run is always slower)
        _ = app.get(test_img)

        # Timed runs
        times = []
        for i in range(10):
            start = time.time()
            faces = app.get(test_img)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed*1000:.1f}ms ({len(faces)} faces detected)")

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\n✓ Average face detection time: {avg_time*1000:.1f}ms")
        print(f"  Min: {min_time*1000:.1f}ms, Max: {max_time*1000:.1f}ms")
        print(f"  Max FPS (if detecting every frame): {1/avg_time:.1f} FPS")

        # Performance assessment
        if avg_time < 0.05:  # Less than 50ms
            print(f"✓ Performance: EXCELLENT (CoreML + Neural Engine working)")
        elif avg_time < 0.1:  # Less than 100ms
            print(f"⚠️  Performance: GOOD but could be better")
        elif avg_time < 0.2:  # Less than 200ms
            print(f"⚠️  Performance: SLOW - Check CoreML provider")
        else:
            print(f"✗ Performance: VERY SLOW - CoreML likely not working")
            print(f"  Expected: ~20-30ms on M1 with CoreML")
            print(f"  Actual: {avg_time*1000:.1f}ms")

        return avg_time

    except Exception as e:
        print(f"✗ Error testing face detection: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_face_swapper_speed():
    """Test face swapper inference speed"""
    print_section("FACE SWAPPER INFERENCE TEST")

    try:
        import numpy as np
        import onnxruntime as ort

        models_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")

        if not os.path.exists(model_path):
            print(f"✗ Model not found: {model_path}")
            return None

        print(f"Loading model: {model_path}")

        # Get providers
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")

        # Create session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Setup provider options for M1
        if 'CoreMLExecutionProvider' in providers:
            provider_list = [
                ('CoreMLExecutionProvider', {
                    'compute_units': 'CPU_AND_NE',
                    'allow_low_precision': True,
                }),
                ('CPUExecutionProvider', {})
            ]
            print("✓ Using CoreML with Neural Engine")
        else:
            provider_list = [('CPUExecutionProvider', {})]
            print("⚠️  Using CPU only (will be SLOW)")

        # Load model
        start = time.time()
        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=[p[0] for p in provider_list]
        )
        load_time = time.time() - start
        print(f"✓ Model loaded in {load_time:.2f}s")

        # Get input info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"  Input: {input_name}, Shape: {input_shape}")

        # Create test input
        test_input = np.random.rand(1, 3, 128, 128).astype(np.float32)

        # Warm-up
        print("\nWarming up model...")
        _ = session.run(None, {input_name: test_input})

        # Timed runs
        print("Testing inference speed...")
        times = []
        for i in range(10):
            start = time.time()
            output = session.run(None, {input_name: test_input})
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed*1000:.1f}ms")

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\n✓ Average inference time: {avg_time*1000:.1f}ms")
        print(f"  Min: {min_time*1000:.1f}ms, Max: {max_time*1000:.1f}ms")
        print(f"  Max FPS (inference only): {1/avg_time:.1f} FPS")

        # Performance assessment
        if avg_time < 0.03:  # Less than 30ms
            print(f"✓ Performance: EXCELLENT (Neural Engine working)")
        elif avg_time < 0.05:  # Less than 50ms
            print(f"⚠️  Performance: GOOD")
        elif avg_time < 0.1:  # Less than 100ms
            print(f"⚠️  Performance: ACCEPTABLE but slow")
        else:
            print(f"✗ Performance: VERY SLOW")
            print(f"  Expected: ~15-25ms on M1 with Neural Engine")
            print(f"  Actual: {avg_time*1000:.1f}ms")

        return avg_time

    except Exception as e:
        print(f"✗ Error testing face swapper: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_optimizations_active():
    """Check if optimizations are in the code"""
    print_section("OPTIMIZATION CODE CHECK")

    checks = {
        "modules/processors/frame/face_swapper_optimized.py": [
            ("M1 chip detection", "IS_M1_CHIP = _detect_m1_chip()"),
            ("Face caching (90 frames)", "face_detection_interval = 90"),
            ("Neural Engine provider", "CPU_AND_NE"),
            ("Adaptive frame skipping", "AdaptiveFrameSkipper"),
            ("Thread pool optimization", "max_workers = 2"),
        ],
        "modules/performance_optimizer.py": [
            ("Contiguous memory", "_initialize_pool_contiguous"),
            ("NEON optimization", "OPENCV_ENABLE_NEON"),
        ],
        "modules/apple_silicon_config.py": [
            ("M1 resolution presets", "720.*540"),
        ],
    }

    for file_path, markers in checks.items():
        if os.path.exists(file_path):
            print(f"\n✓ {file_path}")
            with open(file_path, 'r') as f:
                content = f.read()
                for name, marker in markers:
                    if marker in content:
                        print(f"  ✓ {name}: Found")
                    else:
                        print(f"  ✗ {name}: NOT FOUND")
        else:
            print(f"\n✗ {file_path}: File not found")


def estimate_expected_fps():
    """Estimate expected FPS"""
    print_section("EXPECTED PERFORMANCE BREAKDOWN")

    print("\nExpected M1 Performance (with ALL optimizations):")
    print("=" * 50)
    print("  Face detection (cached):      ~2-3ms per frame")
    print("  Face swapping (Neural Engine): ~20-25ms per frame")
    print("  Image operations (NEON):       ~3-5ms per frame")
    print("  Overhead:                      ~2-3ms per frame")
    print("  " + "-" * 48)
    print("  Total per frame:               ~30-40ms")
    print("  Expected FPS:                  25-33 FPS")
    print()
    print("Balanced mode (720×540):         24-28 FPS target")
    print("Performance mode (640×480):      28-32 FPS target")
    print()
    print("If you're getting 0.8 FPS (~1250ms per frame):")
    print("  You are 30-40x SLOWER than expected!")
    print()
    print("Common causes (check above results):")
    print("  1. CoreML provider not available (CPU fallback)")
    print("  2. Face detection not cached (running every frame)")
    print("  3. Wrong script running (run_m3_optimized.py required)")
    print("  4. Models corrupted or missing")


def main():
    """Run all diagnostics"""
    print("\n" + "=" * 70)
    print(" MAC M1 PERFORMANCE DIAGNOSTIC TOOL")
    print(" Current Performance: 0.8 FPS")
    print(" Expected Performance: 24-28 FPS")
    print("=" * 70)
    print("\nThis script will identify why performance is degraded.")
    print("Estimated time: 30-60 seconds")

    # Run all checks
    check_system_info()
    check_python_packages()
    check_onnx_providers()
    models_ok = check_models()

    if models_ok:
        face_det_time = test_face_detection_speed()
        face_swap_time = test_face_swapper_speed()
    else:
        print("\n⚠️  Skipping performance tests (models not found)")
        face_det_time = None
        face_swap_time = None

    check_optimizations_active()
    estimate_expected_fps()

    # Final summary
    print_section("DIAGNOSTIC SUMMARY")

    issues = []

    # Check CoreML
    try:
        import onnxruntime as ort
        if 'CoreMLExecutionProvider' not in ort.get_available_providers():
            issues.append("❌ CRITICAL: CoreML provider not available")
    except:
        issues.append("❌ CRITICAL: Cannot import onnxruntime")

    # Check models
    if not models_ok:
        issues.append("❌ CRITICAL: Models missing or corrupted")

    # Check performance
    if face_det_time and face_det_time > 0.1:
        issues.append("⚠️  Face detection is slow (>100ms)")

    if face_swap_time and face_swap_time > 0.05:
        issues.append("⚠️  Face swapping is slow (>50ms)")

    if issues:
        print("\n🔴 ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ All checks passed!")
        print("  If you're still getting 0.8 FPS, the issue may be:")
        print("  - Running wrong script (use run_m3_optimized.py)")
        print("  - Optimizations not loading at runtime")
        print("  - Face detection not being cached")

    print("\n" + "=" * 70)
    print(" NEXT STEPS")
    print("=" * 70)
    print("\n1. Review the results above")
    print("2. Fix any CRITICAL issues first")
    print("3. Ensure you run: python run_m3_optimized.py")
    print("4. Check console output for M1 detection messages")
    print("5. Monitor FPS during first 30 seconds of running")
    print()
    print("For detailed troubleshooting, see: TROUBLESHOOTING_LOW_FPS.md")


if __name__ == "__main__":
    main()
