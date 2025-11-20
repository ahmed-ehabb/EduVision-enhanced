#!/usr/bin/env python3
"""
Simple Camera Test Script
Diagnoses camera access issues for the distraction detection system
"""

import cv2
import sys
import time

def test_camera():
    """Test camera access and display basic info"""
    print("🔍 Testing Camera Access...")
    print("-" * 40)
    
    try:
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ ERROR: Cannot access camera")
            print("   Possible causes:")
            print("   • Camera is being used by another application")
            print("   • Camera permissions denied")
            print("   • No camera connected")
            print("   • Camera driver issues")
            return False
        
        print("✅ Camera device opened successfully")
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📹 Camera Info:")
        print(f"   • Resolution: {width}x{height}")
        print(f"   • FPS: {fps}")
        
        # Try to read a frame
        print("\n📸 Testing frame capture...")
        ret, frame = cap.read()
        
        if not ret:
            print("❌ ERROR: Cannot read frames from camera")
            cap.release()
            return False
        
        print("✅ Frame captured successfully")
        print(f"   • Frame shape: {frame.shape}")
        
        # Test multiple frame reads
        print("\n🎬 Testing continuous capture (5 frames)...")
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"   Frame {i+1}: ✅ {frame.shape}")
            else:
                print(f"   Frame {i+1}: ❌ Failed")
            time.sleep(0.1)
        
        cap.release()
        print("\n🎉 Camera test completed successfully!")
        print("   Your camera is working and ready for detection.")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_opencv_installation():
    """Test OpenCV installation"""
    print("\n🔧 Testing OpenCV Installation...")
    print("-" * 40)
    
    try:
        print(f"✅ OpenCV Version: {cv2.__version__}")
        
        # Test basic OpenCV functions
        import numpy as np
        test_array = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_array, cv2.COLOR_BGR2GRAY)
        print("✅ OpenCV color conversion works")
        
        return True
    except Exception as e:
        print(f"❌ OpenCV Error: {str(e)}")
        return False

def main():
    print("🚀 Camera Diagnostic Tool")
    print("=" * 50)
    
    # Test OpenCV first
    opencv_ok = test_opencv_installation()
    
    if not opencv_ok:
        print("\n❌ OpenCV issues detected. Please reinstall:")
        print("   pip install opencv-python")
        return False
    
    # Test camera
    camera_ok = test_camera()
    
    print("\n" + "=" * 50)
    if camera_ok:
        print("🎯 DIAGNOSIS: Camera is working correctly!")
        print("✅ You can now run the distraction detection system")
        print("\n💡 Next steps:")
        print("   1. Start web interface: python start_web_ui.py")
        print("   2. Click 'Start Detection' in the web dashboard")
        print("   3. Or run directly: python run_local_cv.py")
    else:
        print("🔧 DIAGNOSIS: Camera issues detected!")
        print("\n🛠️ Troubleshooting steps:")
        print("   1. Close any apps using the camera (Zoom, Teams, etc.)")
        print("   2. Check camera permissions in Windows Settings")
        print("   3. Try unplugging and reconnecting USB camera")
        print("   4. Restart your computer")
        print("   5. Update camera drivers")
    
    return camera_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 