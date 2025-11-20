"""
Test script for EduVision API
Tests authentication flow and endpoints
"""

import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8002"

# Test data
test_teacher = {
    "email": "teacher.test@eduvision.com",
    "password": "TeacherPass123",
    "full_name": "Dr. Jane Smith",
    "role": "teacher",
    "department": "Computer Science"
}

test_student = {
    "email": "student.test@eduvision.com",
    "password": "StudentPass123",
    "full_name": "John Doe",
    "role": "student",
    "student_number": "2021001",
    "major": "Computer Science",
    "year_of_study": 3
}

# Store tokens
teacher_tokens = {}
student_tokens = {}

def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_result(test_name: str, response: requests.Response):
    """Print test result"""
    status_icon = "[OK]" if response.status_code < 300 else "[FAIL]"
    print(f"{status_icon} {test_name}")
    print(f"    Status: {response.status_code}")

    if response.status_code < 300:
        try:
            data = response.json()
            # Print limited response data
            if "access_token" in data:
                print(f"    Access Token: {data['access_token'][:50]}...")
                if "user" in data:
                    print(f"    User: {data['user']['email']} ({data['user']['role']})")
            elif "email" in data:
                print(f"    User: {data['email']}")
            elif "message" in data:
                print(f"    Message: {data['message']}")
        except:
            pass
    else:
        try:
            error = response.json()
            print(f"    Error: {error.get('detail', response.text)}")
        except:
            print(f"    Error: {response.text}")

def test_health_check():
    """Test health check endpoint"""
    print_section("Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print_result("GET /health", response)
    return response.status_code == 200

def test_register_teacher():
    """Test teacher registration"""
    global teacher_tokens
    print_section("Teacher Registration")

    response = requests.post(
        f"{BASE_URL}/auth/register",
        json=test_teacher
    )

    print_result("POST /auth/register (teacher)", response)

    if response.status_code == 201:
        data = response.json()
        teacher_tokens = {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"]
        }
        return True

    return False

def test_register_student():
    """Test student registration"""
    global student_tokens
    print_section("Student Registration")

    response = requests.post(
        f"{BASE_URL}/auth/register",
        json=test_student
    )

    print_result("POST /auth/register (student)", response)

    if response.status_code == 201:
        data = response.json()
        student_tokens = {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"]
        }
        return True

    return False

def test_duplicate_registration():
    """Test duplicate email registration (should fail)"""
    print_section("Duplicate Registration (Should Fail)")

    response = requests.post(
        f"{BASE_URL}/auth/register",
        json=test_teacher  # Try to register same email again
    )

    print_result("POST /auth/register (duplicate)", response)
    return response.status_code == 400

def test_login_teacher():
    """Test teacher login"""
    global teacher_tokens
    print_section("Teacher Login")

    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={
            "email": test_teacher["email"],
            "password": test_teacher["password"]
        }
    )

    print_result("POST /auth/login (teacher)", response)

    if response.status_code == 200:
        data = response.json()
        teacher_tokens = {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"]
        }
        return True

    return False

def test_login_student():
    """Test student login"""
    global student_tokens
    print_section("Student Login")

    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={
            "email": test_student["email"],
            "password": test_student["password"]
        }
    )

    print_result("POST /auth/login (student)", response)

    if response.status_code == 200:
        data = response.json()
        student_tokens = {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"]
        }
        return True

    return False

def test_get_current_user():
    """Test getting current user info"""
    print_section("Get Current User Info")

    # Test with teacher token
    response = requests.get(
        f"{BASE_URL}/auth/me",
        headers={"Authorization": f"Bearer {teacher_tokens['access_token']}"}
    )
    print_result("GET /auth/me (teacher)", response)

    # Test with student token
    response = requests.get(
        f"{BASE_URL}/auth/me",
        headers={"Authorization": f"Bearer {student_tokens['access_token']}"}
    )
    print_result("GET /auth/me (student)", response)

    return response.status_code == 200

def test_refresh_token():
    """Test token refresh"""
    global teacher_tokens
    print_section("Token Refresh")

    response = requests.post(
        f"{BASE_URL}/auth/refresh",
        json={"refresh_token": teacher_tokens["refresh_token"]}
    )

    print_result("POST /auth/refresh", response)

    if response.status_code == 200:
        data = response.json()
        teacher_tokens = {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"]
        }
        return True

    return False

def test_invalid_token():
    """Test authentication with invalid token (should fail)"""
    print_section("Invalid Token (Should Fail)")

    response = requests.get(
        f"{BASE_URL}/auth/me",
        headers={"Authorization": "Bearer invalid.token.here"}
    )

    print_result("GET /auth/me (invalid token)", response)
    return response.status_code == 401

def test_change_password():
    """Test password change"""
    print_section("Change Password")

    response = requests.put(
        f"{BASE_URL}/users/me/password",
        headers={"Authorization": f"Bearer {student_tokens['access_token']}"},
        json={
            "current_password": test_student["password"],
            "new_password": "NewStudentPass456"
        }
    )

    print_result("PUT /users/me/password", response)

    # Try logging in with new password
    if response.status_code == 204:
        login_response = requests.post(
            f"{BASE_URL}/auth/login",
            json={
                "email": test_student["email"],
                "password": "NewStudentPass456"
            }
        )
        print_result("POST /auth/login (new password)", login_response)
        return login_response.status_code == 200

    return False

def test_logout():
    """Test logout (revoke tokens)"""
    print_section("Logout")

    response = requests.post(
        f"{BASE_URL}/auth/logout",
        headers={"Authorization": f"Bearer {teacher_tokens['access_token']}"}
    )

    print_result("POST /auth/logout", response)
    return response.status_code == 204

def run_all_tests():
    """Run all API tests"""
    print("\n" + "=" * 70)
    print("  EduVision API - Authentication Flow Test")
    print("=" * 70)

    tests = [
        ("Health Check", test_health_check),
        ("Register Teacher", test_register_teacher),
        ("Register Student", test_register_student),
        ("Duplicate Registration", test_duplicate_registration),
        ("Login Teacher", test_login_teacher),
        ("Login Student", test_login_student),
        ("Get Current User", test_get_current_user),
        ("Refresh Token", test_refresh_token),
        ("Invalid Token", test_invalid_token),
        ("Change Password", test_change_password),
        ("Logout", test_logout),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[ERROR] {test_name}: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)
    print(f"  Total Tests: {passed + failed}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest error: {e}")
        import traceback
        traceback.print_exc()
