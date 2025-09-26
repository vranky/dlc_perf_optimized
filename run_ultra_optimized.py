#!/usr/bin/env python3
"""
Ultra-Optimized Deep Live Cam Runner for Mac M1/M3
Target: 20+ FPS with memory efficiency
"""

import sys
import os
import platform
import time
import argparse
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import optimization setup
try:
    from setup_apple_silicon import setup_environment_variables, detect_apple_silicon, optimize_python_environment
    
    # Auto-configure for Apple Silicon
    chip_type, total_cores = detect_apple_silicon()
    if chip_type:
        print(f"🚀 Detected {chip_type} - Applying optimizations...")
        setup_environment_variables(chip_type, total_cores)
        optimize_python_environment()
    else:
        print("⚠️  Non-Apple Silicon system detected - using standard optimizations")
        
except ImportError:
    print("⚠️  Could not load Apple Silicon optimizations")

# Import modules
import modules.globals
import modules.metadata
from modules.ui_optimized import create_optimized_ui

def print_system_info():
    """Print system information and optimization status"""
    print("=" * 60)
    print(f"🎯 Deep Live Cam Ultra-Optimized v{modules.metadata.version}")
    print(f"🖥️  System: {platform.system()} {platform.release()}")
    print(f"🔧 Processor: {platform.processor()}")
    
    # Detect Apple Silicon details
    chip_type, total_cores = detect_apple_silicon()
    if chip_type:
        print(f"🚀 Apple Silicon: {chip_type} ({total_cores} cores)")
        print(f"🎯 Target Performance: 20+ FPS")
        
        # Check memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            print(f"🧠 Unified Memory: {memory_gb:.1f} GB")
        except ImportError:
            pass
    
    print("=" * 60)

def setup_modules():
    """Setup modules with optimal configuration"""
    
    # Set execution providers for Apple Silicon
    if platform.system() == 'Darwin' and platform.processor() == 'arm':
        modules.globals.execution_providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    else:
        modules.globals.execution_providers = ['CPUExecutionProvider']
    
    # Default paths
    modules.globals.source_path = None
    modules.globals.target_path = None  
    modules.globals.output_path = None
    
    # Performance settings
    modules.globals.frame_processors = ['face_swapper_optimized']
    modules.globals.keep_fps = True
    modules.globals.keep_audio = True
    modules.globals.many_faces = False
    modules.globals.map_faces = False
    modules.globals.nsfw_filter = False
    modules.globals.live_mirror = True
    modules.globals.live_resizable = False
    
    # Memory optimization settings
    modules.globals.max_memory = 8 * 1024 * 1024 * 1024  # 8GB limit
    modules.globals.execution_thread_count = 8  # Use performance cores

def check_model_availability():
    """Check if required models are available"""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    required_models = [
        "inswapper_128_fp16.onnx"
    ]
    
    missing_models = []
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if not os.path.exists(model_path):
            missing_models.append(model)
    
    if missing_models:
        print(f"❌ Missing models: {', '.join(missing_models)}")
        print(f"📁 Models directory: {models_dir}")
        print("💡 Download models and place them in the models directory")
        return False
    
    print("✅ All required models found")
    return True

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Ultra-Optimized Deep Live Cam")
    parser.add_argument('--enable-monitoring', action='store_true', 
                       help='Enable performance monitoring')
    parser.add_argument('--memory-limit', type=int, default=8,
                       help='Memory limit in GB (default: 8)')
    parser.add_argument('--face-detection-interval', type=int, default=30,
                       help='Face detection interval in frames (default: 30)')
    
    args = parser.parse_args()
    
    print_system_info()
    
    # Check model availability
    if not check_model_availability():
        sys.exit(1)
    
    # Setup modules
    setup_modules()
    
    # Apply command line arguments
    if args.memory_limit:
        modules.globals.max_memory = args.memory_limit * 1024 * 1024 * 1024
    
    print("🎮 Starting Ultra-Optimized UI...")
    print("💡 Use 'Performance' mode for best FPS on Mac M1")
    print("💡 Use 'Balanced' mode for good quality on Mac M3+")
    
    try:
        # Create and run optimized UI
        ui = create_optimized_ui()
        
        print("✅ Application started successfully")
        print("📊 Monitor FPS and memory usage in the UI")
        
        # Start the main loop
        start_time = time.time()
        ui.run()
        
        # Calculate session time
        session_time = time.time() - start_time
        print(f"🕒 Session duration: {session_time:.1f} seconds")
        
    except KeyboardInterrupt:
        print("\n👋 Application terminated by user")
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        print("🧹 Cleaning up resources...")
        
        # Cleanup face swapper
        try:
            from modules.optimized_face_swapper_v2 import _face_swapper_instance
            if _face_swapper_instance:
                _face_swapper_instance.cleanup()
        except ImportError:
            pass
        
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()