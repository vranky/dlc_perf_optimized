#!/usr/bin/env python3
"""
Test script to verify face_swapper_optimized module can be imported and loaded correctly
"""

import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all critical imports work"""
    print("Testing imports...")
    
    try:
        # Test face_types import
        from modules.face_types import Face, Frame
        print("✓ face_types import successful")
        
        # Test performance_optimizer import 
        from modules.performance_optimizer import (
            FPSMonitor, FrameBufferPool, AppleSiliconOptimizer, 
            BatchProcessor, PerformanceMetrics
        )
        print("✓ performance_optimizer import successful")
        
        # Test face_swapper_optimized import
        import modules.processors.frame.face_swapper_optimized as fso
        print("✓ face_swapper_optimized import successful")
        print(f"  Module name: {fso.NAME}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Other error: {e}")
        return False

def test_frame_processor_loading():
    """Test the frame processor loading mechanism"""
    print("\nTesting frame processor loading...")
    
    try:
        import modules.globals
        
        # Initialize required globals
        modules.globals.frame_processors = ['face_swapper_optimized']
        modules.globals.execution_providers = ['CPUExecutionProvider']
        modules.globals.fp_ui = {'face_enhancer': False, 'face_swapper': False}
        
        # Test the frame processor loading
        from modules.processors.frame.core import get_frame_processors_modules
        
        processors = get_frame_processors_modules(['face_swapper_optimized'])
        print(f"✓ Successfully loaded {len(processors)} processor(s)")
        
        for processor in processors:
            print(f"  - {processor.NAME}")
            
            # Verify all required methods exist
            required_methods = ['pre_check', 'pre_start', 'process_frame', 'process_image', 'process_video']
            all_methods_present = True
            
            for method in required_methods:
                if hasattr(processor, method) and callable(getattr(processor, method)):
                    print(f"    ✓ {method}")
                else:
                    print(f"    ✗ {method} missing")
                    all_methods_present = False
                    
            if all_methods_present:
                print("    ✓ All required methods present")
                
        return True
        
    except Exception as e:
        print(f"✗ Frame processor loading failed: {e}")
        return False

def test_both_processors():
    """Test that both standard and optimized processors work together"""
    print("\nTesting compatibility with standard face_swapper...")
    
    try:
        import modules.globals
        modules.globals.frame_processors = ['face_swapper', 'face_swapper_optimized']
        modules.globals.execution_providers = ['CPUExecutionProvider']
        modules.globals.fp_ui = {'face_enhancer': False}
        
        from modules.processors.frame.core import get_frame_processors_modules
        
        # Clear the global cache to test fresh loading
        import modules.processors.frame.core as core
        core.FRAME_PROCESSORS_MODULES = []
        
        processors = get_frame_processors_modules(['face_swapper', 'face_swapper_optimized'])
        print(f"✓ Successfully loaded both processors")
        
        processor_names = [p.NAME for p in processors]
        print(f"  Loaded: {processor_names}")
        
        expected_names = ['DLC.FACE-SWAPPER', 'DLC.FACE-SWAPPER-OPTIMIZED']
        if all(name in str(processor_names) for name in expected_names):
            print("✓ Both processors loaded correctly")
            return True
        else:
            print("✗ Not all expected processors loaded")
            return False
            
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing face_swapper_optimized module")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_frame_processor_loading,
        test_both_processors
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
            
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The face_swapper_optimized module is working correctly.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)