"""
Quick API Test - Uses Short Audio
==================================

Fast API test using a short audio file for quick verification.

Author: Ahmed
Date: 2025-11-06
"""

import requests
import json
import time
from pathlib import Path

API_BASE_URL = "http://localhost:8000"
TEST_DIR = Path(__file__).parent
TEST_AUDIO = TEST_DIR / "model_tests" / "test_audio.wav"  # Short test audio

def test_quick():
    """Quick test with short audio."""
    print("\n" + "="*80)
    print("  QUICK API TEST")
    print("="*80)

    # Check server
    print("\n[1] Checking server health...")
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"    Status: {resp.json()['status']}")
    except Exception as e:
        print(f"    ERROR: Server not running - {e}")
        return

    # Upload lecture
    print("\n[2] Uploading short test audio...")

    textbook_paras = [
        "Psychology is the scientific study of mind and behavior.",
        "Cognitive processes include attention, memory, and perception.",
        "Learning involves changes in behavior due to experience."
    ]

    files = {'audio_file': ('test.wav', open(TEST_AUDIO, 'rb'), 'audio/wav')}
    data = {
        'textbook_paragraphs': json.dumps(textbook_paras),
        'lecture_title': 'Quick Test Lecture',
        'subject': 'Psychology',
        'teacher_id': 'test_teacher'
    }

    try:
        resp = requests.post(f"{API_BASE_URL}/api/lectures/analyze", files=files, data=data, timeout=30)
        if resp.status_code != 200:
            print(f"    ERROR: {resp.status_code} - {resp.text}")
            return

        result = resp.json()
        lecture_id = result['lecture_id']
        print(f"    Lecture ID: {lecture_id}")
        print(f"    Status: {result['status']}")
    finally:
        files['audio_file'][1].close()

    # Poll for results
    print("\n[3] Waiting for processing...")
    print("    (Short audio should complete in ~2-3 minutes)")

    start = time.time()
    max_wait = 300  # 5 minutes max

    while time.time() - start < max_wait:
        try:
            resp = requests.get(f"{API_BASE_URL}/api/lectures/{lecture_id}", timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                status = result.get('status')

                if status == 'completed':
                    elapsed = time.time() - start
                    print(f"\n    COMPLETED in {elapsed:.1f} seconds!")

                    print("\n[4] Results Summary:")
                    if 'transcript' in result:
                        print(f"    - Transcript: {len(result['transcript'].get('text', ''))} chars")
                    if 'engagement' in result:
                        print(f"    - Engagement: {result['engagement'].get('overall_score', 0):.1f}%")
                    if 'content_alignment' in result:
                        print(f"    - Coverage: {result['content_alignment'].get('coverage_score', 0):.1f}%")
                    if 'notes' in result:
                        print(f"    - Notes: {len(result['notes'].get('bullet_points', []))} points")
                    if 'quiz' in result:
                        print(f"    - Quiz: {len(result['quiz'].get('questions', []))} questions")

                    print("\n" + "="*80)
                    print("  SUCCESS - API is working correctly!")
                    print("="*80)
                    return

                elif status == 'failed':
                    print(f"\n    FAILED: {result.get('error')}")
                    return

                else:
                    print(f"\r    Status: {status} ({result.get('progress', 0)}%)...", end='', flush=True)

        except Exception as e:
            print(f"\r    Polling error: {e}...", end='', flush=True)

        time.sleep(5)

    print(f"\n    TIMEOUT after {max_wait} seconds")

if __name__ == "__main__":
    test_quick()
