#!/usr/bin/env python3
"""
Camera Detection Script
This script helps you identify which camera indices are available on your system.
"""

import cv2

def detect_cameras():
    """Detect all available cameras and their properties."""
    print("Detecting available cameras...")
    print("-" * 50)
    
    available_cameras = []
    
    # Test camera indices 0-9 (usually sufficient for most systems)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera {i}: AVAILABLE")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            
            # Try to read a frame to verify it's working
            ret, frame = cap.read()
            if ret:
                print(f"  Status: Working ✓")
                available_cameras.append(i)
            else:
                print(f"  Status: Connected but not responding ✗")
            
            cap.release()
            print()
        else:
            print(f"Camera {i}: Not available")
    
    print("-" * 50)
    if available_cameras:
        print(f"Available cameras: {available_cameras}")
        print(f"Use camera index {available_cameras[0]} as default")
    else:
        print("No cameras detected!")
    
    return available_cameras

if __name__ == "__main__":
    detect_cameras()
