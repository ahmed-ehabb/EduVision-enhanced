"""
Test Quiz Model Loading
=======================

Simple script to test if the quiz model can load successfully.

Author: Ahmed
Date: 2025-11-06
"""

import sys
import os
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("\n" + "="*80)
print("  TESTING QUIZ MODEL LOADING")
print("="*80)

print("\n[1] Importing quiz generator...")
try:
    from quiz_generator_v2 import QuizGeneratorV2
    print("    ✓ Import successful")
except Exception as e:
    print(f"    ✗ Import failed: {e}")
    sys.exit(1)

print("\n[2] Initializing quiz generator...")
try:
    quiz_gen = QuizGeneratorV2()
    print("    ✓ Initialization successful")
except Exception as e:
    print(f"    ✗ Initialization failed: {e}")
    sys.exit(1)

print("\n[3] Loading model (this is the critical step)...")
print("    This may take 30-60 seconds...")
print("    Watch for 'Loading checkpoint shards' progress...")

start_time = time.time()

try:
    quiz_gen.load_model()
    elapsed = time.time() - start_time
    print(f"    ✓ Model loaded successfully in {elapsed:.1f} seconds!")
except Exception as e:
    elapsed = time.time() - start_time
    print(f"    ✗ Model loading failed after {elapsed:.1f} seconds")
    print(f"    Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4] Testing quiz generation with simple text...")
test_text = """
Psychology is the scientific study of mind and behavior.
Cognitive processes include attention, memory, perception, and reasoning.
Memory formation involves encoding, storage, and retrieval of information.
"""

try:
    result = quiz_gen.generate_quiz(test_text, num_mcq=2, num_open_ended=1)
    print(f"    ✓ Quiz generation successful!")
    print(f"    Generated {len(result.get('questions', []))} questions")
except Exception as e:
    print(f"    ✗ Quiz generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[5] Cleaning up...")
try:
    quiz_gen.cleanup()
    print("    ✓ Cleanup successful")
except Exception as e:
    print(f"    ⚠ Cleanup warning: {e}")

print("\n" + "="*80)
print("  TEST COMPLETE")
print("="*80)
print("\nIf all steps passed, the quiz model is working correctly.")
print("The issue is likely with the API server threading/async context.")
print("\n" + "="*80)
