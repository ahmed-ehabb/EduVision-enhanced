"""
Test Worker Direct Execution
=============================

Test that the pipeline worker can be called as a subprocess.

Author: Ahmed
Date: 2025-11-06
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

# Test configuration
TEST_DIR = Path(__file__).parent
TEST_AUDIO = TEST_DIR / "testing" / "model_tests" / "test_audio.wav"

def test_worker_subprocess():
    """Test calling worker as subprocess."""
    print("=" * 80)
    print("  TESTING WORKER SUBPROCESS")
    print("=" * 80)

    # Create test config
    temp_dir = tempfile.gettempdir()
    config_file = os.path.join(temp_dir, "test_worker_config.json")
    results_file = os.path.join(temp_dir, "test_worker_results.json")

    config = {
        "lecture_id": "test-worker-123",
        "audio_path": str(TEST_AUDIO.absolute()),
        "textbook_paragraphs": [
            "Psychology is the scientific study of mind and behavior.",
            "Cognitive processes include attention and memory.",
            "Learning involves changes in behavior."
        ],
        "lecture_title": "Test Worker Lecture",
        "pdf_path": None,
        "results_file": results_file
    }

    print(f"\n[1] Writing test config to: {config_file}")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print("    Config written")

    print(f"\n[2] Calling worker subprocess...")
    worker_script = TEST_DIR / "backend" / "pipeline_worker.py"
    print(f"    Worker: {worker_script}")
    print(f"    Python: {sys.executable}")

    try:
        result = subprocess.run(
            [sys.executable, str(worker_script), config_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        print(f"\n[3] Worker completed with exit code: {result.returncode}")

        if result.stdout:
            print(f"\n[STDOUT]:")
            print(result.stdout)

        if result.stderr:
            print(f"\n[STDERR]:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"\n[4] Checking results file...")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)

                if results.get('success'):
                    print("    ✓ Pipeline succeeded!")
                    print(f"    - Transcript: {len(results.get('transcript', {}).get('text', ''))} chars")
                    print(f"    - Quiz: {len(results.get('quiz', {}).get('questions', []))} questions")
                else:
                    print("    ✗ Pipeline failed:")
                    for error in results.get('errors', []):
                        print(f"      - {error}")

                # Clean up
                os.unlink(results_file)
            else:
                print("    ✗ No results file created")
        else:
            print(f"\n    ✗ Worker failed with exit code {result.returncode}")

        # Clean up config
        if os.path.exists(config_file):
            os.unlink(config_file)

        print("\n" + "=" * 80)
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("\n    ✗ Worker timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"\n    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_worker_subprocess()
    sys.exit(0 if success else 1)
