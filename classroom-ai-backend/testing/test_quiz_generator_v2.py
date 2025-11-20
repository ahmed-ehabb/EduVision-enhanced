"""
Test Quiz Generator V2 - Production Deployment Test

Tests the deployed Quiz Generator V2 with:
1. Model loading
2. RAG retrieval
3. MCQ generation
4. End-to-end quiz generation
"""

import sys
import os
import time
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from quiz_generator_v2 import QuizGeneratorV2, create_quiz_generator


def print_section(title: str):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_model_loading():
    """Test 1: Model Loading"""
    print_section("TEST 1: Model Loading")

    try:
        generator = QuizGeneratorV2()

        print("[INFO] Loading model with 4-bit quantization...")
        start_time = time.time()

        generator.load_model()

        load_time = time.time() - start_time

        print(f"[OK] Model loaded successfully in {load_time:.2f}s")

        # Get model info
        info = generator.get_model_info()
        print("\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Cleanup
        generator.cleanup()

        return True

    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        return False


def test_rag_retrieval():
    """Test 2: RAG Retrieval"""
    print_section("TEST 2: RAG Retrieval")

    try:
        generator = QuizGeneratorV2()
        generator.load_model()

        print("[INFO] Loading RAG components (lightweight mode)...")
        generator.load_rag_components(pdf_path=None)  # Use lightweight KB

        print("[OK] RAG components loaded")

        # Test retrieval
        test_lecture = """
        Classical conditioning is a type of learning discovered by Ivan Pavlov.
        It involves pairing a neutral stimulus with an unconditioned stimulus.
        """

        print("\n[INFO] Testing context retrieval...")
        context = generator.retrieve_context(test_lecture, top_k=3)

        print(f"[OK] Retrieved {len(context)} context chunks")
        print("\nRetrieved Context:")
        for i, chunk in enumerate(context, 1):
            print(f"  {i}. {chunk[:100]}...")

        # Cleanup
        generator.cleanup()

        return len(context) > 0

    except Exception as e:
        print(f"[FAIL] RAG retrieval failed: {e}")
        return False


def test_mcq_generation():
    """Test 3: MCQ Generation"""
    print_section("TEST 3: MCQ Generation")

    try:
        generator = QuizGeneratorV2()
        generator.load_model()
        generator.load_rag_components(pdf_path=None)

        # Sample context
        context = """
        Classical conditioning is a learning process where a neutral stimulus becomes
        associated with a meaningful stimulus. Ivan Pavlov demonstrated this with dogs
        that learned to salivate at the sound of a bell.
        """

        print("[INFO] Generating MCQ questions...")
        start_time = time.time()

        mcq_output = generator.generate_mcq_questions(
            context=context,
            num_questions=2,
            max_new_tokens=300
        )

        gen_time = time.time() - start_time

        print(f"[OK] Generation completed in {gen_time:.2f}s")
        print("\nGenerated Output:")
        print("-" * 80)
        print(mcq_output)
        print("-" * 80)

        # Check if output looks reasonable
        has_question = "?" in mcq_output or "What" in mcq_output or "Who" in mcq_output
        has_options = any(letter in mcq_output for letter in ["a)", "b)", "c)", "d)"])

        if has_question and has_options:
            print("\n[OK] Output contains question and options")
        else:
            print("\n[!] Warning: Output may not be properly formatted")

        # Cleanup
        generator.cleanup()

        return has_question and has_options

    except Exception as e:
        print(f"[FAIL] MCQ generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test 4: Full Pipeline (End-to-End)"""
    print_section("TEST 4: Full Pipeline (End-to-End)")

    try:
        # Sample lecture
        lecture = """
        Today we're discussing operant conditioning, a learning theory developed by
        B.F. Skinner. Operant conditioning involves learning through consequences.
        When a behavior is followed by a reward, it is more likely to be repeated.
        This is called reinforcement. When a behavior is followed by punishment,
        it is less likely to be repeated. Skinner demonstrated these principles
        using a device called the Skinner box, where animals could press a lever
        to receive food rewards.
        """

        print("[INFO] Creating quiz generator...")
        generator = create_quiz_generator()  # Uses factory function

        print("\n[INFO] Generating quiz from lecture...")
        start_time = time.time()

        result = generator.generate_quiz_from_lecture(
            lecture_text=lecture,
            num_questions=3,
            max_new_tokens=500
        )

        total_time = time.time() - start_time

        print(f"[OK] Quiz generated in {total_time:.2f}s")

        # Display results
        print("\n" + "="*80)
        print("Generated Quiz Questions:")
        print("="*80)

        questions = result.get("questions", [])

        if len(questions) == 0:
            print("\n[!] Warning: No questions were parsed from output")
            print("\nRaw Output:")
            print(result.get("raw_output", "No output"))

        for i, q in enumerate(questions, 1):
            print(f"\n{i}. {q.get('question', 'N/A')}")
            for opt in q.get('options', []):
                print(f"   {opt}")
            print(f"   Correct answer: {q.get('correct_answer', 'N/A')}")

        # Metadata
        print("\n" + "="*80)
        print("Metadata:")
        print("="*80)
        metadata = result.get("metadata", {})
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        # Cleanup
        generator.cleanup()

        return metadata.get("success", False)

    except Exception as e:
        print(f"[FAIL] Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """Test 5: Memory Usage Monitoring"""
    print_section("TEST 5: Memory Usage Monitoring")

    try:
        import torch

        if not torch.cuda.is_available():
            print("[!] CUDA not available - skipping memory test")
            return True

        # Initial memory
        torch.cuda.empty_cache()
        initial_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print(f"Initial GPU Memory: {initial_allocated:.2f}GB / {total_memory:.2f}GB")

        # Load model
        generator = QuizGeneratorV2()
        generator.load_model()

        # Memory after model load
        after_load = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"After Model Load: {after_load:.2f}GB / {total_memory:.2f}GB")
        print(f"Model Memory Usage: {after_load - initial_allocated:.2f}GB")

        # Generate quiz
        generator.load_rag_components(pdf_path=None)
        lecture = "Classical conditioning is a learning process discovered by Pavlov."

        result = generator.generate_quiz_from_lecture(lecture, num_questions=2)

        # Memory after generation
        after_gen = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"After Generation: {after_gen:.2f}GB / {total_memory:.2f}GB")

        # Cleanup
        generator.cleanup()

        # Memory after cleanup
        torch.cuda.empty_cache()
        after_cleanup = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"After Cleanup: {after_cleanup:.2f}GB / {total_memory:.2f}GB")

        # Check if within 4GB limit
        max_usage = max(initial_allocated, after_load, after_gen)
        within_limit = max_usage <= 4.0

        print(f"\n[{'OK' if within_limit else 'FAIL'}] Max usage: {max_usage:.2f}GB (Limit: 4.00GB)")

        return within_limit

    except Exception as e:
        print(f"[FAIL] Memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("  QUIZ GENERATOR V2 - PRODUCTION DEPLOYMENT TEST SUITE")
    print("="*80)

    tests = [
        ("Model Loading", test_model_loading),
        ("RAG Retrieval", test_rag_retrieval),
        ("MCQ Generation", test_mcq_generation),
        ("Full Pipeline", test_full_pipeline),
        ("Memory Usage", test_memory_usage),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            print(f"\n\nRunning: {test_name}...")
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n[FAIL] Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80 + "\n")

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for test_name, result in results.items():
        status = "[OK PASS]" if result else "[FAIL FAIL]"
        print(f"  {status} {test_name}")

    print("\n" + "="*80)
    print(f"  TOTAL: {passed}/{total} tests passed")
    print("="*80)

    if passed == total:
        print("\n[OK] All tests passed! Quiz Generator V2 is ready for production.")
    else:
        print(f"\n[!] {total - passed} test(s) failed. Please review errors above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
