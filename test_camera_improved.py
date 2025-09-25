#!/usr/bin/env python3
"""
Test improved camera access with macOS-specific initialization
"""
import cv2
import time
import platform

def test_improved_camera():
    print("Testing improved camera access...")
    print(f"Platform: {platform.system()}")
    
    cap = None
    
    if platform.system() == "Darwin":  # macOS
        print("Using macOS-specific camera initialization...")
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        for backend in backends:
            try:
                print(f"Trying backend: {backend}")
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    # Force read a frame to ensure camera initialization
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Camera initialized successfully with backend {backend}")
                        print(f"Frame shape: {frame.shape}")
                        break
                    else:
                        print(f"❌ Camera opened but can't read frames with backend {backend}")
                        cap.release()
                        cap = None
                else:
                    print(f"❌ Failed to open camera with backend {backend}")
            except Exception as e:
                print(f"❌ Error with backend {backend}: {e}")
                continue
    else:
        cap = cv2.VideoCapture(0)
    
    if not cap or not cap.isOpened():
        print("❌ Failed to initialize camera")
        return False
    
    # Configure format
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera properties after configuration:")
    print(f"  Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Give camera time to initialize hardware on macOS
    if platform.system() == "Darwin":
        print("Waiting for camera hardware initialization...")
        time.sleep(0.5)
        print("Hardware initialization delay completed")
    
    # Try to read frames
    print("Reading test frames...")
    for i in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  Frame {i+1}: SUCCESS - {frame.shape}")
        else:
            print(f"  Frame {i+1}: FAILED to read")
        time.sleep(0.1)
    
    cap.release()
    print("Camera test completed")
    return True

if __name__ == "__main__":
    test_improved_camera()