"""
Student Module Verification Script
===================================

Comprehensive verification of all Student Module components.

Usage:
    python verify_setup.py
"""

import sys
import os
from pathlib import Path

print("=" * 80)
print("STUDENT MODULE VERIFICATION")
print("=" * 80)
print()

# Test results tracking
tests_passed = 0
tests_failed = 0
warnings = []

def test_result(name, passed, details=""):
    """Print and track test result."""
    global tests_passed, tests_failed
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {name}")
    if details:
        print(f"         {details}")
    if passed:
        tests_passed += 1
    else:
        tests_failed += 1
    print()

# ============================================================================
# TEST 1: Python Version
# ============================================================================
print("TEST 1: Python Version")
print("-" * 80)
python_version = sys.version_info
version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
is_compatible = python_version >= (3, 11)
test_result(
    "Python Version",
    is_compatible,
    f"Version {version_str} (Required: 3.11+)"
)

# ============================================================================
# TEST 2: Required Packages
# ============================================================================
print("TEST 2: Required Packages")
print("-" * 80)

packages_to_check = [
    ("fastapi", "0.104.1"),
    ("uvicorn", "0.24.0"),
    ("opencv-python", "4.8.1.78"),
    ("opencv-contrib-python", "4.8.1.78"),
    ("mediapipe", "0.10.8"),
    ("tensorflow", "2.15.0"),
    ("numpy", "1.24.3"),
    ("pillow", "10.1.0"),
]

for package_name, expected_version in packages_to_check:
    try:
        if package_name == "opencv-python":
            import cv2
            installed_version = cv2.__version__
            package_obj = cv2
        elif package_name == "opencv-contrib-python":
            import cv2
            # Check for cv2.face module
            has_face = hasattr(cv2, 'face')
            test_result(
                f"{package_name}",
                has_face,
                f"cv2.face module: {'Available' if has_face else 'MISSING - install opencv-contrib-python'}"
            )
            continue
        elif package_name == "pillow":
            # Pillow is imported as PIL
            from PIL import Image
            installed_version = Image.__version__ if hasattr(Image, '__version__') else "Unknown"
            module = Image
        else:
            module = __import__(package_name.replace("-", "_"))
            installed_version = module.__version__
            package_obj = module

        version_match = installed_version.startswith(expected_version.split('.')[0])
        test_result(
            f"{package_name}",
            True,
            f"Version {installed_version}"
        )
    except ImportError as e:
        test_result(f"{package_name}", False, f"Not installed: {e}")
    except Exception as e:
        test_result(f"{package_name}", False, f"Error: {e}")

# ============================================================================
# TEST 3: Directory Structure
# ============================================================================
print("TEST 3: Directory Structure")
print("-" * 80)

required_dirs = [
    ("facedataset/train", True),
    ("util/model", True),
    ("frontend_outputs/student", False),
    ("frontend_outputs/teacher", False),
    ("templates", False),
    ("static", False),
]

for dir_path, critical in required_dirs:
    full_path = Path(dir_path)
    exists = full_path.exists() and full_path.is_dir()

    if not exists and not critical:
        warnings.append(f"Optional directory missing: {dir_path}")

    test_result(
        f"Directory: {dir_path}",
        exists or not critical,
        f"{'Exists' if exists else 'Missing' + (' (optional)' if not critical else ' (REQUIRED)')}"
    )

# ============================================================================
# TEST 4: Model Files
# ============================================================================
print("TEST 4: Model Files")
print("-" * 80)

model_files = [
    "util/model/emotion_recognition.h5",
    "util/model/emotion_best_weights.h5",
    "util/model/haarcascade_frontalface_default.xml",
    "util/model/shape_predictor_68_face_landmarks.dat",
]

for model_path in model_files:
    full_path = Path(model_path)
    exists = full_path.exists() and full_path.is_file()
    size_mb = full_path.stat().st_size / (1024 * 1024) if exists else 0

    test_result(
        f"Model: {Path(model_path).name}",
        exists,
        f"{'Size: {:.1f} MB'.format(size_mb) if exists else 'MISSING'}"
    )

# ============================================================================
# TEST 5: Face Dataset
# ============================================================================
print("TEST 5: Face Recognition Dataset")
print("-" * 80)

dataset_path = Path("facedataset/train")
if dataset_path.exists():
    students = [d for d in dataset_path.iterdir() if d.is_dir()]
    total_images = 0

    for student_dir in students:
        images = list(student_dir.glob("*.jpg")) + list(student_dir.glob("*.png"))
        total_images += len(images)
        test_result(
            f"Student: {student_dir.name}",
            len(images) > 0,
            f"{len(images)} training images"
        )

    if total_images == 0:
        warnings.append("No training images found in facedataset/train/")
else:
    test_result("Dataset directory", False, "facedataset/train not found")

# ============================================================================
# TEST 6: Configuration Files
# ============================================================================
print("TEST 6: Configuration Files")
print("-" * 80)

config_files = [
    "fastapi_ui.py",
    "requirements.txt",
    "Dockerfile",
    "util/config.py",
    "util/identity_verifier.py",
]

for config_file in config_files:
    full_path = Path(config_file)
    exists = full_path.exists() and full_path.is_file()
    test_result(f"File: {config_file}", exists, "Found" if exists else "MISSING")

# ============================================================================
# TEST 7: Import Test - Core Modules
# ============================================================================
print("TEST 7: Core Module Imports")
print("-" * 80)

try:
    from util import config as config_module
    test_result("util.config", True, f"NO_FACE_SECS={config_module.NO_FACE_SECS}")
except Exception as e:
    test_result("util.config", False, str(e))

try:
    from util.identity_verifier import IDVerifier
    test_result("IDVerifier", True, "Class imported successfully")
except Exception as e:
    test_result("IDVerifier", False, str(e))

try:
    import cv2
    # Test LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    test_result("LBPH Face Recognizer", True, "cv2.face.LBPHFaceRecognizer_create() works")
except Exception as e:
    test_result("LBPH Face Recognizer", False, str(e))

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    test_result("MediaPipe Face Mesh", True, "MediaPipe solutions available")
except Exception as e:
    test_result("MediaPipe Face Mesh", False, str(e))

try:
    import tensorflow as tf
    # Suppress TF warnings for test
    tf.get_logger().setLevel('ERROR')
    test_result("TensorFlow", True, f"Version {tf.__version__}")
except Exception as e:
    test_result("TensorFlow", False, str(e))

# ============================================================================
# TEST 8: API Server Test
# ============================================================================
print("TEST 8: API Server Configuration")
print("-" * 80)

try:
    # Check if fastapi_ui.py has correct port
    with open("fastapi_ui.py", "r", encoding="utf-8") as f:
        content = f.read()
        has_port_8001 = "port=8001" in content or 'port = 8001' in content
        has_health_endpoint = '@app.get("/health")' in content

        test_result("Port Configuration", has_port_8001, "Port 8001 configured" if has_port_8001 else "Port NOT set to 8001")
        test_result("Health Endpoint", has_health_endpoint, "/health endpoint exists" if has_health_endpoint else "/health endpoint MISSING")
except Exception as e:
    test_result("API Configuration Check", False, str(e))

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
print(f"Tests Passed:  {tests_passed}")
print(f"Tests Failed:  {tests_failed}")
print(f"Warnings:      {len(warnings)}")
print()

if warnings:
    print("[WARNING] Warnings found:")
    for warning in warnings:
        print(f"  - {warning}")
    print()

if tests_failed == 0:
    print("[OK] ALL TESTS PASSED - Student Module is ready!")
    print()
    print("Next steps:")
    print("  1. Start the API server:")
    print("     .venv/Scripts/python.exe fastapi_ui.py")
    print()
    print("  2. Test the health endpoint:")
    print("     curl http://localhost:8001/health")
    print()
    print("  3. Access the dashboard:")
    print("     http://localhost:8001")
    sys.exit(0)
else:
    print(f"[ERROR] {tests_failed} TEST(S) FAILED - Please fix the issues above")
    sys.exit(1)
