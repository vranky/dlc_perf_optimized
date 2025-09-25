#!/usr/bin/env python3
"""
Test all available cameras to find working ones
"""
import cv2
import time

def test_all_cameras():
    print("Testing all available camera indices...")
    
    working_cameras = []
    
    # Test camera indices 0-5
    for camera_id in range(6):
        print(f"\n--- Testing Camera {camera_id} ---")
        
        # Test different backends
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        
        for backend_id, backend in enumerate(backends):
            backend_name = "CAP_AVFOUNDATION" if backend == cv2.CAP_AVFOUNDATION else "CAP_ANY"
            print(f"  Backend: {backend_name}")
            
            try:
                cap = cv2.VideoCapture(camera_id, backend)
                
                if cap.isOpened():
                    print(f"    ✅ Camera {camera_id} opened with {backend_name}")
                    
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"    ✅ Successfully read frame: {frame.shape}")
                        working_cameras.append((camera_id, backend, frame.shape))
                        
                        # Read a few more frames to test stability
                        success_count = 0
                        for i in range(5):
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                success_count += 1
                            time.sleep(0.1)
                        
                        print(f"    📊 Read {success_count}/5 additional frames successfully")
                        cap.release()
                        break  # Found working backend for this camera
                    else:
                        print(f"    ❌ Camera {camera_id} opened but failed to read frame")
                        cap.release()
                else:
                    print(f"    ❌ Failed to open camera {camera_id} with {backend_name}")
                    
            except Exception as e:
                print(f"    ❌ Exception with camera {camera_id}, backend {backend_name}: {e}")
    
    print(f"\n{'='*50}")
    print("SUMMARY:")
    if working_cameras:
        print("Working cameras found:")
        for camera_id, backend, shape in working_cameras:
            backend_name = "CAP_AVFOUNDATION" if backend == cv2.CAP_AVFOUNDATION else "CAP_ANY"
            print(f"  Camera {camera_id} with {backend_name}: {shape}")
    else:
        print("❌ No working cameras found")
    
    return working_cameras

if __name__ == "__main__":
    test_all_cameras()