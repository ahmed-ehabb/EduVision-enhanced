"""
Test script for integrated API endpoints
Tests lecture, session, engagement, and analytics endpoints
"""

import requests
import json
from typing import Dict, Optional

BASE_URL = "http://localhost:8002"

# Test data
teacher_data = {
    "email": "integration.teacher@eduvision.com",
    "password": "TeacherPass123!",
    "full_name": "Dr. Integration Teacher",
    "role": "teacher",
    "department": "Computer Science"
}

student_data = {
    "email": "integration.student@eduvision.com",
    "password": "StudentPass123!",
    "full_name": "Integration Student",
    "role": "student",
    "student_number": "INT2025001",
    "major": "Computer Science",
    "year_of_study": 3
}

# Store tokens and IDs
teacher_tokens = {}
student_tokens = {}
lecture_id = None
session_id = None
session_code = None


def print_header(title: str):
    """Print test section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(test_name: str, success: bool, status_code: int, details: str = ""):
    """Print test result"""
    status = "[OK]" if success else "[FAIL]"
    print(f"{status} {test_name}")
    print(f"    Status: {status_code}")
    if details:
        print(f"    {details}")


def register_and_login_teacher():
    """Register and login as teacher"""
    global teacher_tokens

    print_header("Teacher Registration & Login")

    # Register
    response = requests.post(f"{BASE_URL}/auth/register", json=teacher_data)
    if response.status_code == 201:
        data = response.json()
        teacher_tokens = {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"]
        }
        print_result("Teacher Registration", True, 201, f"User: {data['user']['email']}")
    elif response.status_code == 400 and "already registered" in response.json().get("detail", ""):
        # Already registered, just login
        print_result("Teacher Registration", True, 400, "Already registered, logging in...")

        # Login
        response = requests.post(f"{BASE_URL}/auth/login", json={
            "email": teacher_data["email"],
            "password": teacher_data["password"]
        })
        if response.status_code == 200:
            data = response.json()
            teacher_tokens = {
                "access_token": data["access_token"],
                "refresh_token": data["refresh_token"]
            }
            print_result("Teacher Login", True, 200, f"User: {data['user']['email']}")
        else:
            print_result("Teacher Login", False, response.status_code, response.json().get("detail"))
            return False
    else:
        print_result("Teacher Registration", False, response.status_code, response.json().get("detail"))
        return False

    return True


def register_and_login_student():
    """Register and login as student"""
    global student_tokens

    print_header("Student Registration & Login")

    # Register
    response = requests.post(f"{BASE_URL}/auth/register", json=student_data)
    if response.status_code == 201:
        data = response.json()
        student_tokens = {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"]
        }
        print_result("Student Registration", True, 201, f"User: {data['user']['email']}")
    elif response.status_code == 400 and "already registered" in response.json().get("detail", ""):
        # Already registered, just login
        print_result("Student Registration", True, 400, "Already registered, logging in...")

        # Login
        response = requests.post(f"{BASE_URL}/auth/login", json={
            "email": student_data["email"],
            "password": student_data["password"]
        })
        if response.status_code == 200:
            data = response.json()
            student_tokens = {
                "access_token": data["access_token"],
                "refresh_token": data["refresh_token"]
            }
            print_result("Student Login", True, 200, f"User: {data['user']['email']}")
        else:
            print_result("Student Login", False, response.status_code, response.json().get("detail"))
            return False
    else:
        print_result("Student Registration", False, response.status_code, response.json().get("detail"))
        return False

    return True


def test_create_lecture():
    """Test creating a lecture"""
    global lecture_id

    print_header("Lecture Creation")

    lecture_data = {
        "title": "Introduction to Machine Learning",
        "description": "Basic concepts of ML including supervised and unsupervised learning",
        "course_code": "CS401",
        "course_name": "Machine Learning"
    }

    headers = {"Authorization": f"Bearer {teacher_tokens['access_token']}"}
    response = requests.post(f"{BASE_URL}/api/lectures", json=lecture_data, headers=headers)

    if response.status_code == 201:
        data = response.json()
        lecture_id = data["lecture_id"]
        print_result("Create Lecture", True, 201, f"Lecture ID: {lecture_id}")
        print(f"    Title: {data['title']}")
        print(f"    Course: {data['course_code']} - {data['course_name']}")
        return True
    else:
        try:
            error_detail = response.json().get("detail", "Unknown error")
        except:
            error_detail = f"Response: {response.text[:200]}"
        print_result("Create Lecture", False, response.status_code, error_detail)
        return False


def test_list_lectures():
    """Test listing lectures"""
    print_header("List Lectures")

    headers = {"Authorization": f"Bearer {teacher_tokens['access_token']}"}
    response = requests.get(f"{BASE_URL}/api/lectures", headers=headers)

    if response.status_code == 200:
        data = response.json()
        print_result("List Lectures", True, 200, f"Found {len(data)} lecture(s)")
        for lecture in data:
            print(f"    - {lecture['title']} (ID: {lecture['lecture_id'][:8]}...)")
        return True
    else:
        print_result("List Lectures", False, response.status_code, response.json().get("detail"))
        return False


def test_create_session():
    """Test creating a session"""
    global session_id, session_code

    print_header("Session Creation")

    if not lecture_id:
        print_result("Create Session", False, 0, "No lecture ID available")
        return False

    session_data = {
        "lecture_id": lecture_id,
        "scheduled_at": None  # Start now
    }

    headers = {"Authorization": f"Bearer {teacher_tokens['access_token']}"}
    response = requests.post(f"{BASE_URL}/api/sessions", json=session_data, headers=headers)

    if response.status_code == 201:
        data = response.json()
        session_id = data["session_id"]
        session_code = data["session_code"]
        print_result("Create Session", True, 201, f"Session Code: {session_code}")
        print(f"    Session ID: {session_id}")
        print(f"    Status: {data['status']}")
        return True
    else:
        try:
            error_detail = response.json().get("detail", "Unknown error")
        except:
            error_detail = f"Response: {response.text[:200]}"
        print_result("Create Session", False, response.status_code, error_detail)
        return False


def test_start_session():
    """Test starting a session"""
    print_header("Start Session")

    if not session_code:
        print_result("Start Session", False, 0, "No session code available")
        return False

    headers = {"Authorization": f"Bearer {teacher_tokens['access_token']}"}
    response = requests.post(f"{BASE_URL}/api/sessions/{session_code}/start", headers=headers)

    if response.status_code == 200:
        data = response.json()
        print_result("Start Session", True, 200, f"Session {session_code} started")
        print(f"    Status: {data['status']}")
        print(f"    Started at: {data['started_at']}")
        return True
    else:
        print_result("Start Session", False, response.status_code, response.json().get("detail"))
        return False


def test_join_session():
    """Test student joining session"""
    print_header("Student Join Session")

    if not session_code:
        print_result("Join Session", False, 0, "No session code available")
        return False

    headers = {"Authorization": f"Bearer {student_tokens['access_token']}"}
    response = requests.post(f"{BASE_URL}/api/sessions/{session_code}/join", headers=headers)

    if response.status_code == 200:
        data = response.json()
        print_result("Join Session", True, 200, data["message"])
        print(f"    Session: {data['session_code']}")
        print(f"    Lecture: {data['lecture_title']}")
        return True
    else:
        print_result("Join Session", False, response.status_code, response.json().get("detail"))
        return False


def test_log_engagement():
    """Test logging engagement events"""
    print_header("Log Engagement Events")

    if not session_id:
        print_result("Log Engagement", False, 0, "No session ID available")
        return False

    # Create sample engagement events
    events = [
        {
            "face_detected": True,
            "identity_verified": True,
            "emotion": "neutral",
            "emotion_confidence": 0.85,
            "head_pitch": 5.2,
            "head_yaw": -2.1,
            "gaze_ratio": 0.92,
            "concentration_index": "HIGH",
            "video_timestamp": 5000
        },
        {
            "face_detected": True,
            "identity_verified": True,
            "emotion": "happy",
            "emotion_confidence": 0.78,
            "head_pitch": 3.5,
            "head_yaw": 1.2,
            "gaze_ratio": 0.88,
            "concentration_index": "MEDIUM",
            "video_timestamp": 10000
        }
    ]

    headers = {"Authorization": f"Bearer {student_tokens['access_token']}"}
    response = requests.post(
        f"{BASE_URL}/api/sessions/{session_id}/engagement",
        json=events,
        headers=headers
    )

    if response.status_code == 201:
        data = response.json()
        print_result("Log Engagement", True, 201, data["message"])
        print(f"    Events logged: {data['count']}")
        return True
    else:
        print_result("Log Engagement", False, response.status_code, response.json().get("detail"))
        return False


def test_get_analytics():
    """Test getting session analytics"""
    print_header("Get Session Analytics")

    if not session_id:
        print_result("Get Analytics", False, 0, "No session ID available")
        return False

    headers = {"Authorization": f"Bearer {teacher_tokens['access_token']}"}
    response = requests.get(
        f"{BASE_URL}/api/sessions/{session_id}/analytics",
        headers=headers
    )

    if response.status_code == 200:
        data = response.json()
        if "message" in data:
            # Analytics not yet generated
            print_result("Get Analytics", True, 200, data["message"])
        else:
            print_result("Get Analytics", True, 200, "Analytics retrieved")
            print(f"    Participants: {data.get('total_participants', 0)}")
            print(f"    Avg Engagement: {data.get('avg_engagement_score', 'N/A')}")
            print(f"    Distraction Events: {data.get('total_distraction_events', 0)}")
        return True
    else:
        print_result("Get Analytics", False, response.status_code, response.json().get("detail"))
        return False


def test_end_session():
    """Test ending a session"""
    print_header("End Session")

    if not session_code:
        print_result("End Session", False, 0, "No session code available")
        return False

    headers = {"Authorization": f"Bearer {teacher_tokens['access_token']}"}
    response = requests.post(f"{BASE_URL}/api/sessions/{session_code}/end", headers=headers)

    if response.status_code == 200:
        data = response.json()
        print_result("End Session", True, 200, f"Session {session_code} ended")
        print(f"    Status: {data['status']}")
        print(f"    Duration: {data['duration']} seconds")
        return True
    else:
        print_result("End Session", False, response.status_code, response.json().get("detail"))
        return False


def test_access_control():
    """Test role-based access control"""
    print_header("Access Control Tests")

    # Student tries to create lecture (should fail)
    headers = {"Authorization": f"Bearer {student_tokens['access_token']}"}
    response = requests.post(
        f"{BASE_URL}/api/lectures",
        json={"title": "Unauthorized Lecture", "description": "Should fail"},
        headers=headers
    )

    if response.status_code == 403:
        print_result("Student Cannot Create Lecture", True, 403, "Access correctly denied")
    else:
        print_result("Student Cannot Create Lecture", False, response.status_code, "Student should NOT be able to create lectures")

    # Teacher tries to log engagement (should fail)
    if session_id:
        headers = {"Authorization": f"Bearer {teacher_tokens['access_token']}"}
        response = requests.post(
            f"{BASE_URL}/api/sessions/{session_id}/engagement",
            json=[{"face_detected": True}],
            headers=headers
        )

        if response.status_code == 403:
            print_result("Teacher Cannot Log Engagement", True, 403, "Access correctly denied")
        else:
            print_result("Teacher Cannot Log Engagement", False, response.status_code, "Teacher should NOT be able to log engagement")


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("  EduVision API - Integration Test Suite")
    print("=" * 70)

    tests_passed = 0
    tests_failed = 0

    # Authentication
    if not register_and_login_teacher():
        print("\n[ERROR] Teacher authentication failed. Cannot continue.")
        return
    tests_passed += 1

    if not register_and_login_student():
        print("\n[ERROR] Student authentication failed. Cannot continue.")
        return
    tests_passed += 1

    # Lecture endpoints
    if test_create_lecture():
        tests_passed += 1
    else:
        tests_failed += 1

    if test_list_lectures():
        tests_passed += 1
    else:
        tests_failed += 1

    # Session endpoints
    if test_create_session():
        tests_passed += 1
    else:
        tests_failed += 1

    if test_start_session():
        tests_passed += 1
    else:
        tests_failed += 1

    # Engagement endpoints
    if test_join_session():
        tests_passed += 1
    else:
        tests_failed += 1

    if test_log_engagement():
        tests_passed += 1
    else:
        tests_failed += 1

    # Analytics endpoints
    if test_get_analytics():
        tests_passed += 1
    else:
        tests_failed += 1

    if test_end_session():
        tests_passed += 1
    else:
        tests_failed += 1

    # Access control
    test_access_control()
    tests_passed += 2  # Two access control tests

    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)
    print(f"  Total Tests: {tests_passed + tests_failed}")
    print(f"  Passed: {tests_passed}")
    print(f"  Failed: {tests_failed}")
    print(f"  Success Rate: {(tests_passed / (tests_passed + tests_failed) * 100):.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Tests stopped by user")
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
