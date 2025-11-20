"""
Teacher Module V2 API Test Script
==================================

Tests the REST API endpoints for the Teacher Module V2.

Usage:
    python testing/test_teacher_api.py

Requirements:
    - API server running on http://localhost:8000
    - Test audio file and textbook content

Author: Ahmed (with AI assistance)
Date: 2025-11-06
"""

import os
import sys
import time
import requests
from pathlib import Path

# API configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 300  # 5 minutes timeout for long operations

# Test data paths
TEST_DIR = Path(__file__).parent
TEST_AUDIO = TEST_DIR / "lecture_1.mp3"  # Use existing test audio
TEST_TEXTBOOK_PDF = TEST_DIR.parent / "Psychology2e_WEB.pdf"  # Real textbook PDF


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_result(test_name, success, details=""):
    """Print test result."""
    status = "[PASS]" if success else "[FAIL]"
    print(f"\n{status} {test_name}")
    if details:
        print(f"     {details}")


def test_health_check():
    """Test 1: Health check endpoint."""
    print_header("TEST 1: Health Check")

    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"\nStatus: {data.get('status')}")
            print(f"Services: {data.get('services')}")
            print(f"Version: {data.get('version')}")

            success = data.get('status') == 'healthy'
            print_result("Health Check", success)
            return success
        else:
            print_result("Health Check", False, f"HTTP {response.status_code}")
            return False

    except Exception as e:
        print_result("Health Check", False, str(e))
        return False


def test_api_status():
    """Test 2: API status endpoint."""
    print_header("TEST 2: API Status")

    try:
        response = requests.get(f"{API_BASE_URL}/api/status", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"\nStatus: {data.get('status')}")
            print(f"Version: {data.get('version')}")
            print(f"Statistics: {data.get('statistics')}")
            print(f"Features: {data.get('features')}")

            success = data.get('status') == 'operational'
            print_result("API Status", success)
            return success
        else:
            print_result("API Status", False, f"HTTP {response.status_code}")
            return False

    except Exception as e:
        print_result("API Status", False, str(e))
        return False


def test_lecture_analysis():
    """Test 3: Lecture analysis endpoint."""
    print_header("TEST 3: Lecture Analysis (POST /api/lectures/analyze)")

    # Check if test audio exists
    if not TEST_AUDIO.exists():
        print_result("Lecture Analysis", False, f"Test audio not found: {TEST_AUDIO}")
        print("\nNote: Run test_teacher_module_e2e.py first to generate test audio")
        return False

    try:
        print(f"\nUploading audio: {TEST_AUDIO.name}")
        print("Textbook: Using default Psychology2e_WEB.pdf (auto-loaded by API)")

        # Prepare multipart form data
        # NOTE: Not sending textbook_paragraphs - API will auto-load from Psychology2e_WEB.pdf
        files = {
            'audio_file': ('test_lecture.mp3', open(TEST_AUDIO, 'rb'), 'audio/mpeg')
        }
        data = {
            'lecture_title': 'Test Psychology Lecture',
            'subject': 'Psychology',
            'teacher_id': 'test_teacher_001'
        }

        print("\nSending request to API...")
        response = requests.post(
            f"{API_BASE_URL}/api/lectures/analyze",
            files=files,
            data=data,
            timeout=30
        )

        files['audio_file'][1].close()  # Close file

        if response.status_code == 200:
            result = response.json()
            lecture_id = result.get('lecture_id')

            print(f"\nLecture ID: {lecture_id}")
            print(f"Status: {result.get('status')}")
            print(f"Message: {result.get('message')}")

            print_result("Lecture Analysis", True, f"Lecture queued: {lecture_id}")
            return lecture_id

        else:
            print(f"\nResponse: {response.text}")
            print_result("Lecture Analysis", False, f"HTTP {response.status_code}")
            return None

    except Exception as e:
        print_result("Lecture Analysis", False, str(e))
        return None


def test_lecture_status(lecture_id):
    """Test 4: Lecture status endpoint."""
    print_header("TEST 4: Lecture Status (GET /api/lectures/{id})")

    if not lecture_id:
        print_result("Lecture Status", False, "No lecture_id provided")
        return False

    try:
        print(f"\nLecture ID: {lecture_id}")
        print("\nPolling for status updates...")

        max_wait = 900  # 15 minutes (increased for long processing)
        poll_interval = 5  # 5 seconds
        start_time = time.time()

        while True:
            try:
                response = requests.get(
                    f"{API_BASE_URL}/api/lectures/{lecture_id}",
                    timeout=60  # Increased timeout to 60 seconds
                )
            except requests.exceptions.Timeout:
                # Retry on timeout
                print(f"\n[TIMEOUT] Request timed out, retrying...", end='', flush=True)
                time.sleep(poll_interval)
                continue
            except requests.exceptions.ConnectionError:
                # Retry on connection error
                print(f"\n[ERROR] Connection error, retrying...", end='', flush=True)
                time.sleep(poll_interval)
                continue

            if response.status_code == 200:
                result = response.json()
                status = result.get('status')
                progress = result.get('progress', 0)
                current_step = result.get('current_step', 'N/A')

                print(f"\r[{status.upper()}] Progress: {progress}% - {current_step}     ", end='', flush=True)

                # If completed, show full results
                if status == 'completed':
                    print("\n\nProcessing completed successfully!")

                    print("\n[TRANSCRIPT]")
                    transcript = result.get('transcript', {})
                    if transcript:
                        print(f"  Text length: {len(transcript.get('text', ''))} characters")
                        print(f"  Duration: {transcript.get('duration', 0):.1f} seconds")
                        print(f"  Language: {transcript.get('language', 'N/A')}")
                        print(f"  Confidence: {transcript.get('confidence', 0):.2f}")

                    print("\n[ENGAGEMENT]")
                    engagement = result.get('engagement', {})
                    if engagement:
                        print(f"  Overall Score: {engagement.get('overall_score', 0):.1f}%")
                        print(f"  Label: {engagement.get('overall_label', 'N/A')}")

                    print("\n[CONTENT ALIGNMENT]")
                    alignment = result.get('content_alignment', {})
                    if alignment:
                        print(f"  Coverage Score: {alignment.get('coverage_score', 0):.1f}%")
                        print(f"  Feedback: {alignment.get('feedback', 'N/A')}")

                    print("\n[NOTES]")
                    notes = result.get('notes', {})
                    if notes:
                        bullet_points = notes.get('bullet_points', [])
                        print(f"  Bullet Points: {len(bullet_points)}")
                        for i, point in enumerate(bullet_points[:3], 1):
                            print(f"    {i}. {point}")
                        if len(bullet_points) > 3:
                            print(f"    ... and {len(bullet_points) - 3} more")

                    print("\n[QUIZ]")
                    quiz = result.get('quiz', {})
                    if quiz:
                        questions = quiz.get('questions', [])
                        print(f"  Questions: {len(questions)}")

                    print(f"\nProcessing Time: {result.get('processing_time', 0):.1f} seconds")

                    print_result("Lecture Status", True, "Results retrieved successfully")
                    return True

                # If failed, show error
                elif status == 'failed':
                    print("\n\nProcessing failed!")
                    print(f"Error: {result.get('error', 'Unknown error')}")
                    print_result("Lecture Status", False, "Processing failed")
                    return False

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > max_wait:
                    print(f"\n\nTimeout after {max_wait} seconds")
                    print_result("Lecture Status", False, "Timeout waiting for completion")
                    return False

                # Continue polling
                time.sleep(poll_interval)

            else:
                print(f"\nHTTP {response.status_code}: {response.text}")
                print_result("Lecture Status", False, f"HTTP {response.status_code}")
                return False

    except Exception as e:
        print(f"\n")
        print_result("Lecture Status", False, str(e))
        return False


def test_list_lectures():
    """Test 5: List lectures endpoint."""
    print_header("TEST 5: List Lectures (GET /api/lectures)")

    try:
        response = requests.get(
            f"{API_BASE_URL}/api/lectures",
            params={'limit': 10, 'offset': 0},
            timeout=60  # Increased timeout
        )

        if response.status_code == 200:
            result = response.json()

            print(f"\nTotal lectures: {result.get('total')}")
            print(f"Returned: {result.get('count')}")

            lectures = result.get('lectures', [])
            for lecture in lectures[:5]:  # Show first 5
                print(f"\n  - {lecture.get('lecture_title')}")
                print(f"    ID: {lecture.get('lecture_id')}")
                print(f"    Status: {lecture.get('status')}")
                print(f"    Created: {lecture.get('created_at')}")

            if len(lectures) > 5:
                print(f"\n  ... and {len(lectures) - 5} more")

            print_result("List Lectures", True, f"Found {len(lectures)} lectures")
            return True

        else:
            print_result("List Lectures", False, f"HTTP {response.status_code}")
            return False

    except Exception as e:
        print_result("List Lectures", False, str(e))
        return False


def main():
    """Run all API tests."""
    print("\n" + "=" * 80)
    print("  TEACHER MODULE V2 API TEST SUITE")
    print("=" * 80)
    print(f"\nAPI Base URL: {API_BASE_URL}")
    print(f"Test Audio: {TEST_AUDIO}")
    print(f"Test Directory: {TEST_DIR}")

    # Check if server is running
    print("\n[CHECK] Verifying API server is running...")
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("\n[ERROR] API server not responding!")
            print("Please start the server with: python backend/teacher_api.py")
            return 1
        print("[OK] API server is running")
    except Exception as e:
        print(f"\n[ERROR] Cannot connect to API server: {e}")
        print("Please start the server with: python backend/teacher_api.py")
        return 1

    # Run tests
    results = []

    # Test 1: Health Check
    results.append(("Health Check", test_health_check()))

    # Test 2: API Status
    results.append(("API Status", test_api_status()))

    # Test 3: Lecture Analysis
    lecture_id = test_lecture_analysis()
    results.append(("Lecture Analysis", lecture_id is not None))

    # Test 4: Lecture Status (only if Test 3 succeeded)
    if lecture_id:
        results.append(("Lecture Status", test_lecture_status(lecture_id)))
    else:
        print_header("TEST 4: Lecture Status")
        print_result("Lecture Status", False, "Skipped (Test 3 failed)")
        results.append(("Lecture Status", False))

    # Test 5: List Lectures
    results.append(("List Lectures", test_list_lectures()))

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ({100*passed//total}%)")
    print(f"Failed: {total - passed} ({100*(total-passed)//total}%)")

    print("\nDetailed Results:")
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {test_name}")

    print("\n" + "=" * 80)

    if passed == total:
        print("[SUCCESS] All API tests passed!")
        print("=" * 80)
        return 0
    else:
        print(f"[PARTIAL] {total - passed} test(s) failed")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
