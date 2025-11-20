"""
Test Translation Model - RTX 3050 Deployment Validation

Tests the GGUF-based translation model for:
1. Model loading (CPU/GPU)
2. Arabic to English translation
3. English passthrough
4. Code-switched text handling
5. Memory usage monitoring

Model: ahmedheakl/arazn-llama3-english-gguf (4-bit GGUF)
"""

import sys
import os
import time
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from translation_module import TranslationModel, translate_text, calculate_language_stats, LLAMA_CPP_AVAILABLE


def print_section(title: str):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_dependencies():
    """Test 0: Check Dependencies"""
    print_section("TEST 0: Dependency Check")

    try:
        if not LLAMA_CPP_AVAILABLE:
            print("[FAIL] llama-cpp-python not available")
            print("[INFO] Translation requires llama-cpp-python with GPU support")
            print("[INFO] Install with: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
            return False

        print("[OK] llama-cpp-python is available")

        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"[OK] GPU detected: {gpu_name}")
                print(f"[OK] VRAM: {vram:.2f}GB")
            else:
                print("[WARNING] No GPU detected - will run on CPU")
        except ImportError:
            print("[WARNING] PyTorch not available - cannot check GPU")

        return True

    except Exception as e:
        print(f"[FAIL] Dependency check failed: {e}")
        return False


def test_model_loading():
    """Test 1: Model Loading"""
    print_section("TEST 1: Model Loading")

    try:
        print("[INFO] Loading translation model (GGUF format)...")
        start_time = time.time()

        translator = TranslationModel()

        load_time = time.time() - start_time

        print(f"[OK] Model loaded successfully in {load_time:.2f}s")
        print(f"[OK] Model: {translator.model_name}")
        print(f"[OK] Status: {'Available' if translator.is_available() else 'Not Available'}")

        # Check memory if GPU is used
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"[INFO] GPU Memory: {allocated:.2f}GB / {total:.2f}GB")
        except:
            pass

        return translator

    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_arabic_translation(translator):
    """Test 2: Arabic to English Translation"""
    print_section("TEST 2: Arabic to English Translation")

    try:
        # Test with pure Arabic text
        arabic_text = "مرحبا بكم في محاضرة علم النفس اليوم"

        print(f"[INFO] Input text: {arabic_text}")
        start_time = time.time()

        result = translate_text(arabic_text, translator)

        translate_time = time.time() - start_time

        print(f"[OK] Translation completed in {translate_time:.2f}s")
        print(f"\nOriginal: {arabic_text}")
        print(f"Translation: {result['translated_text']}")

        # Check language stats
        stats = result['language_stats']
        print(f"\nLanguage Statistics:")
        print(f"  Arabic: {stats['arabic_percent']:.1f}%")
        print(f"  English: {stats['english_percent']:.1f}%")
        print(f"  Tokens: {stats['total_tokens']}")

        # Validate translation occurred
        if result['translated_text'] and not result.get('error'):
            print("\n[OK] Translation successful")
            return True
        else:
            print("\n[!] Warning: Translation may have issues")
            return False

    except Exception as e:
        print(f"[FAIL] Arabic translation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_english_passthrough(translator):
    """Test 3: English Passthrough"""
    print_section("TEST 3: English Passthrough")

    try:
        # Test with pure English text
        english_text = "Today we will discuss operant conditioning in psychology."

        print(f"[INFO] Input text: {english_text}")
        start_time = time.time()

        result = translate_text(english_text, translator)

        translate_time = time.time() - start_time

        print(f"[OK] Translation completed in {translate_time:.2f}s")
        print(f"\nOriginal: {english_text}")
        print(f"Translation: {result['translated_text']}")

        # Check language stats
        stats = result['language_stats']
        print(f"\nLanguage Statistics:")
        print(f"  English: {stats['english_percent']:.1f}%")
        print(f"  Arabic: {stats['arabic_percent']:.1f}%")

        # English text should have high English percentage
        if stats['english_percent'] > 70:
            print("\n[OK] Correctly identified as English")
            return True
        else:
            print(f"\n[!] Warning: Expected >70% English, got {stats['english_percent']:.1f}%")
            return False

    except Exception as e:
        print(f"[FAIL] English passthrough test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_switched_text(translator):
    """Test 4: Code-Switched Text (Mixed Arabic-English)"""
    print_section("TEST 4: Code-Switched Text Translation")

    try:
        # Test with code-switched text
        mixed_text = "Today's lecture هنتكلم عن operant conditioning وكيف Skinner استخدم reinforcement"

        print(f"[INFO] Input text: {mixed_text}")
        start_time = time.time()

        result = translate_text(mixed_text, translator)

        translate_time = time.time() - start_time

        print(f"[OK] Translation completed in {translate_time:.2f}s")
        print(f"\nOriginal: {mixed_text}")
        print(f"Translation: {result['translated_text']}")

        # Check language stats
        stats = result['language_stats']
        print(f"\nLanguage Statistics:")
        print(f"  English: {stats['english_percent']:.1f}%")
        print(f"  Arabic: {stats['arabic_percent']:.1f}%")
        print(f"  Mixed: {stats['mixed_percent']:.1f}%")

        # Should detect mixed content
        has_both = stats['english_percent'] > 10 and stats['arabic_percent'] > 10

        if has_both:
            print("\n[OK] Correctly identified as code-switched text")
            return True
        else:
            print("\n[!] Warning: Expected mixed Arabic-English")
            return False

    except Exception as e:
        print(f"[FAIL] Code-switched text test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_long_text_handling(translator):
    """Test 5: Long Text Handling (Chunking)"""
    print_section("TEST 5: Long Text Handling")

    try:
        # Create a long text that will require chunking
        long_text = """
        علم النفس هو الدراسة العلمية للسلوك والعقل. يشمل دراسة الظواهر الواعية واللاواعية.
        Psychology is a diverse field that includes many sub-disciplines such as human development,
        sports, health, clinical, social behavior and cognitive processes.
        في هذه المحاضرة سنناقش نظريات التعلم المختلفة.
        We will explore classical conditioning discovered by Pavlov and operant conditioning by Skinner.
        كلا النظريتين مهمتان لفهم كيفية تعلم البشر والحيوانات.
        Classical conditioning involves associating a neutral stimulus with an unconditioned stimulus.
        التكييف الإجرائي يتضمن التعلم من خلال العواقب.
        Reinforcement increases behavior while punishment decreases it.
        هناك أنواع مختلفة من التعزيز والعقاب.
        """ * 3  # Repeat to make it longer

        print(f"[INFO] Input length: {len(long_text.split())} words")

        # Estimate tokens
        estimated_tokens = translator._estimate_tokens(long_text)
        print(f"[INFO] Estimated tokens: {estimated_tokens}")

        start_time = time.time()

        result = translate_text(long_text, translator)

        translate_time = time.time() - start_time

        print(f"[OK] Translation completed in {translate_time:.2f}s")
        print(f"\nTranslation preview (first 200 chars):")
        print(result['translated_text'][:200] + "...")

        print(f"\nProcessing Details:")
        print(f"  Chunks processed: {result.get('chunks_processed', 1)}")
        print(f"  Original length: {result['original_length']} chars")
        print(f"  Translated length: {result['translated_length']} chars")
        print(f"  Processing time: {result['processing_time']:.2f}s")

        # Validate translation occurred
        if result['translated_text'] and not result.get('error'):
            print("\n[OK] Long text handled successfully")
            return True
        else:
            print("\n[!] Warning: Long text handling may have issues")
            return False

    except Exception as e:
        print(f"[FAIL] Long text handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """Test 6: Memory Usage"""
    print_section("TEST 6: Memory Usage Monitoring")

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
        translator = TranslationModel()

        # Memory after model load
        after_load = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"After Model Load: {after_load:.2f}GB / {total_memory:.2f}GB")
        print(f"Model Memory Usage: {after_load - initial_allocated:.2f}GB")

        # Perform translation
        test_text = "مرحبا hello this is a test كلام mixed text"
        result = translate_text(test_text, translator)

        # Memory after translation
        after_translation = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"After Translation: {after_translation:.2f}GB / {total_memory:.2f}GB")

        # Cleanup
        translator = None
        torch.cuda.empty_cache()

        # Memory after cleanup
        after_cleanup = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"After Cleanup: {after_cleanup:.2f}GB / {total_memory:.2f}GB")

        # Check if within 4GB limit
        max_usage = max(initial_allocated, after_load, after_translation)
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
    print("  TRANSLATION MODEL - RTX 3050 DEPLOYMENT TEST SUITE")
    print("="*80)

    # Check dependencies first
    if not test_dependencies():
        print("\n[FAIL] Dependencies not met. Cannot proceed with tests.")
        print("\nTo install llama-cpp-python with GPU support:")
        print("  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
        return False

    # Load model once for most tests
    translator = test_model_loading()

    if not translator:
        print("\n[FAIL] Model loading failed. Cannot proceed with tests.")
        return False

    tests = [
        ("Arabic to English Translation", lambda: test_arabic_translation(translator)),
        ("English Passthrough", lambda: test_english_passthrough(translator)),
        ("Code-Switched Text", lambda: test_code_switched_text(translator)),
        ("Long Text Handling", lambda: test_long_text_handling(translator)),
        ("Memory Usage", test_memory_usage),
    ]

    results = {"Dependency Check": True, "Model Loading": True}

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
        print("\n[OK] All tests passed! Translation model is ready for production.")
    else:
        print(f"\n[!] {total - passed} test(s) failed. Please review errors above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
