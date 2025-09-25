#!/usr/bin/env python3
"""
Simple camera test to verify hardware access
"""
import cv2
import time
import numpy as np

def test_camera():
    print("Testing camera access...")
    
    # Try to open camera 0
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera 0")
        return False
    
    print("✅ Camera 0 opened successfully")
    
    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camera properties:")
    print(f"  Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Try to read a few frames
    print("Reading frames...")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"  Frame {i+1}: {frame.shape if frame is not None else 'None'}")
        else:
            print(f"  Frame {i+1}: Failed to read")
        time.sleep(0.1)
    
    cap.release()
    print("Camera test completed")
    return True

if __name__ == "__main__":
    test_camera()