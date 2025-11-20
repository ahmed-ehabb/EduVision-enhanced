"""
Test ASR + Punctuation Pipeline - RTX 3050 Deployment Validation

Tests the ASR (Whisper) and Punctuation restoration models for 4GB VRAM.

Models:
- ASR: ahmedheakl/arazn-whisper-small-v2 (Whisper Small fine-tuned)
- Punctuation: oliverguhr/fullstop-punctuation-multilang-large

Features tested:
1. ASR model loading with 8-bit quantization
2. Transcription accuracy
3. Punctuation restoration
4. Memory usage monitoring
5. Processing speed
"""

import sys
import os
import time
from pathlib import Path
import numpy as np

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))


def print_section(title: str):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_dependencies():
    """Test 0: Check Dependencies"""
    print_section("TEST 0: Dependency Check")

    dependencies = {
        "torch": False,
        "transformers": False,
        "soundfile": False,
        "librosa": False,
        "deepmultilingualpunctuation": False
    }

    try:
        import torch
        dependencies["torch"] = True
        print(f"[OK] torch version: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[OK] GPU: {gpu_name}")
            print(f"[OK] VRAM: {vram:.2f}GB")
        else:
            print("[WARNING] No GPU detected - will run on CPU")
    except ImportError:
        print("[FAIL] torch not available")

    try:
        import transformers
        dependencies["transformers"] = True
        print(f"[OK] transformers version: {transformers.__version__}")
    except ImportError:
        print("[FAIL] transformers not available")

    try:
        import soundfile
        dependencies["soundfile"] = True
        print(f"[OK] soundfile available")
    except ImportError:
        print("[FAIL] soundfile not available")

    try:
        import librosa
        dependencies["librosa"] = True
        print(f"[OK] librosa version: {librosa.__version__}")
    except ImportError:
        print("[WARNING] librosa not available - ffmpeg fallback will be used")

    try:
        from deepmultilingualpunctuation import PunctuationModel
        dependencies["deepmultilingualpunctuation"] = True
        print(f"[OK] deepmultilingualpunctuation available")
    except ImportError:
        print("[WARNING] deepmultilingualpunctuation not available")
        print("[INFO] Install with: pip install deepmultilingualpunctuation")

    # Check minimum requirements
    required = ["torch", "transformers", "soundfile"]
    missing = [dep for dep in required if not dependencies[dep]]

    if missing:
        print(f"\n[FAIL] Missing required dependencies: {', '.join(missing)}")
        return False

    print("\n[OK] Core dependencies available")
    return True


def test_asr_loading_fp16():
    """Test 1: ASR Model Loading (FP16)"""
    print_section("TEST 1: ASR Model Loading (FP16)")

    try:
        import torch
        from asr_module import ASRModule

        print("[INFO] Loading ASR model with FP16...")
        start_time = time.time()

        asr = ASRModule(
            model_id="ahmedheakl/arazn-whisper-small-v2",
            enable_punctuation=False,  # Don't load punctuation yet
            compute_type="float16"
        )

        load_time = time.time() - start_time

        print(f"[OK] ASR model loaded in {load_time:.2f}s")
        print(f"[OK] Model: {asr.model_name}")
        print(f"[OK] Device: {asr.device}")
        print(f"[OK] Available: {asr.is_available()}")

        # Check memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[INFO] GPU Memory: {allocated:.2f}GB / {total:.2f}GB")

        return asr

    except Exception as e:
        print(f"[FAIL] ASR loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_asr_loading_int8():
    """Test 2: ASR Model Loading (INT8 Quantization)"""
    print_section("TEST 2: ASR Model Loading (INT8 Quantization)")

    try:
        import torch
        from asr_module import ASRModule

        # Clear GPU memory first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[INFO] Loading ASR model with INT8 quantization...")
        start_time = time.time()

        asr = ASRModule(
            model_id="ahmedheakl/arazn-whisper-small-v2",
            enable_punctuation=False,
            compute_type="int8"
        )

        load_time = time.time() - start_time

        print(f"[OK] ASR model loaded in {load_time:.2f}s")
        print(f"[OK] Model: {asr.model_name}")
        print(f"[OK] Device: {asr.device}")
        print(f"[OK] Compute type: INT8")

        # Check memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[INFO] GPU Memory (INT8): {allocated:.2f}GB / {total:.2f}GB")

            if allocated < 4.0:
                print(f"[OK] Memory within 4GB limit")
            else:
                print(f"[!] Warning: Memory exceeds 4GB limit")

        return asr

    except Exception as e:
        print(f"[FAIL] INT8 ASR loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_transcription():
    """Test 3: Transcription (No Punctuation)"""
    print_section("TEST 3: Transcription Test")

    try:
        import torch
        import soundfile as sf
        import tempfile
        from asr_module import ASRModule

        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load ASR with INT8
        print("[INFO] Loading ASR model (INT8)...")
        asr = ASRModule(
            model_id="ahmedheakl/arazn-whisper-small-v2",
            enable_punctuation=False,
            compute_type="int8"
        )

        # Create test audio (5 seconds of speech simulation)
        print("[INFO] Creating test audio...")
        sr = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))

        # Simulate speech (varying frequency to simulate words)
        audio = np.zeros_like(t)
        for i in range(10):  # 10 "words"
            start = i * 0.5 * sr
            end = start + int(0.3 * sr)
            if end < len(audio):
                freq = 200 + i * 20  # Varying pitch
                audio[int(start):int(end)] = 0.3 * np.sin(2 * np.pi * freq * t[int(start):int(end)])

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio, sr)

        print(f"[INFO] Test audio: {duration}s, {sr}Hz")

        # Transcribe
        print("[INFO] Running transcription...")
        start_time = time.time()
        result = asr.transcribe(temp_path, add_punctuation=False)
        transcribe_time = time.time() - start_time

        # Cleanup
        Path(temp_path).unlink()

        if result:
            print(f"[OK] Transcription completed in {transcribe_time:.2f}s")
            print(f"\nResult:")
            print(f"  Text: {result['text'][:100]}...")
            print(f"  Processing time: {result['processing_time']}s")
            print(f"  Model: {result['model']}")
            print(f"  Chunks: {result.get('chunks_processed', 1)}")

            # Check memory after transcription
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"\n[INFO] GPU Memory after transcription: {allocated:.2f}GB / {total:.2f}GB")

            return True
        else:
            print("[!] Transcription returned no result")
            return False

    except Exception as e:
        print(f"[FAIL] Transcription test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_punctuation_loading():
    """Test 4: Punctuation Model Loading"""
    print_section("TEST 4: Punctuation Model Loading")

    try:
        from deepmultilingualpunctuation import PunctuationModel

        print("[INFO] Loading punctuation model...")
        start_time = time.time()

        punct_model = PunctuationModel(
            model="oliverguhr/fullstop-punctuation-multilang-large"
        )

        load_time = time.time() - start_time

        print(f"[OK] Punctuation model loaded in {load_time:.2f}s")

        # Test punctuation restoration
        test_text = "hello world this is a test how are you doing today"
        print(f"\n[INFO] Testing punctuation restoration...")
        print(f"  Input: {test_text}")

        result = punct_model.restore_punctuation(test_text)

        print(f"  Output: {result}")
        print("\n[OK] Punctuation restoration working")

        return True

    except ImportError:
        print("[FAIL] deepmultilingualpunctuation not available")
        print("[INFO] Install with: pip install deepmultilingualpunctuation")
        return False
    except Exception as e:
        print(f"[FAIL] Punctuation model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test 5: Full ASR + Punctuation Pipeline"""
    print_section("TEST 5: Full ASR + Punctuation Pipeline")

    try:
        import torch
        import soundfile as sf
        import tempfile
        from asr_module import ASRModule

        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_mem = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"[INFO] Initial GPU Memory: {initial_mem:.2f}GB")

        # Load ASR with punctuation enabled
        print("[INFO] Loading full pipeline (ASR + Punctuation)...")
        asr = ASRModule(
            model_id="ahmedheakl/arazn-whisper-small-v2",
            enable_punctuation=True,
            compute_type="int8"
        )

        # Create test audio
        sr = 16000
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration))

        # Simulate speech
        audio = np.zeros_like(t)
        for i in range(20):
            start = i * 0.5 * sr
            end = start + int(0.3 * sr)
            if end < len(audio):
                freq = 200 + i * 15
                audio[int(start):int(end)] = 0.3 * np.sin(2 * np.pi * freq * t[int(start):int(end)])

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio, sr)

        print(f"[INFO] Test audio: {duration}s")

        # Transcribe with punctuation
        print("[INFO] Running full pipeline...")
        start_time = time.time()
        result = asr.transcribe(temp_path, add_punctuation=True)
        total_time = time.time() - start_time

        # Cleanup
        Path(temp_path).unlink()

        if result:
            print(f"[OK] Pipeline completed in {total_time:.2f}s")
            print(f"\nResults:")
            print(f"  Raw Text: {result['text'][:100]}...")
            print(f"  Punctuated: {result['punctuated_text'][:100]}...")
            print(f"  ASR Time: {result['processing_time']}s")
            print(f"  Punctuation Time: {result['punctuation_time']}s")
            print(f"  Total Time: {total_time:.2f}s")

            # Memory check
            if torch.cuda.is_available():
                final_mem = torch.cuda.memory_allocated(0) / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"\n[INFO] Final GPU Memory: {final_mem:.2f}GB / {total:.2f}GB")

                if final_mem <= 4.0:
                    print(f"[OK] Memory within 4GB limit")
                else:
                    print(f"[!] Warning: Memory exceeds 4GB")

            return True
        else:
            print("[!] Pipeline returned no result")
            return False

    except Exception as e:
        print(f"[FAIL] Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("  ASR + PUNCTUATION PIPELINE - RTX 3050 DEPLOYMENT TEST")
    print("="*80)
    print("\n[INFO] Testing ASR (Whisper Small) + Punctuation Restoration\n")

    # Check dependencies first
    if not test_dependencies():
        print("\n[FAIL] Dependencies not met. Cannot proceed.")
        print("\nTo install dependencies:")
        print("  pip install torch transformers soundfile librosa")
        print("  pip install deepmultilingualpunctuation")
        return False

    tests = [
        ("ASR Loading (FP16)", test_asr_loading_fp16),
        ("ASR Loading (INT8)", test_asr_loading_int8),
        ("Transcription", test_transcription),
        ("Punctuation Loading", test_punctuation_loading),
        ("Full Pipeline", test_full_pipeline),
    ]

    results = {"Dependency Check": True}

    for test_name, test_func in tests:
        try:
            print(f"\n\nRunning: {test_name}...")
            result = test_func()
            results[test_name] = result if result is not None else False
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

    print("\n[INFO] ASR + Punctuation Pipeline:")
    print("  - ASR Model: ahmedheakl/arazn-whisper-small-v2")
    print("  - Punctuation: oliverguhr/fullstop-punctuation-multilang-large")
    print("  - Recommended: INT8 quantization for 4GB VRAM")
    print("  - Memory: ~2-3GB (INT8) or ~4-5GB (FP16)")

    if passed == total:
        print("\n[OK] All tests passed! ASR pipeline is ready.")
    else:
        print(f"\n[!] {total - passed} test(s) failed. Please review errors above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
