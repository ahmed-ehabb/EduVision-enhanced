"""
Test Quiz Model Loading After Aggressive Cleanup
=================================================

Simulates the pipeline scenario:
1. Load and unload a dummy model (simulate ASR/Notes)
2. Aggressive cleanup
3. Load Quiz model

This tests if the new cleanup strategy prevents segmentation faults.
"""

import sys
import time
import gc
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

import torch


def simulate_previous_model():
    """Simulate loading and unloading a previous model (like Notes)."""
    print("\n" + "="*80)
    print("SIMULATING PREVIOUS MODEL (Notes/ASR)")
    print("="*80)

    # Load a small model to pollute GPU memory
    print("\n[1] Loading dummy model to fragment memory...")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "sentence-transformers/paraphrase-MiniLM-L6-v2"
    ).to("cuda")

    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"    GPU Memory after load: {allocated:.2f}GB")

    # Do some work
    print("[2] Running inference to use memory...")
    time.sleep(1)

    # Unload
    print("[3] Unloading model...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"    GPU Memory after unload: {allocated:.2f}GB")


def aggressive_cleanup():
    """Perform aggressive cleanup (same as in teacher_module_v2.py)."""
    print("\n" + "="*80)
    print("AGGRESSIVE CLEANUP (Same as Pipeline)")
    print("="*80)

    print("\n[1] Double garbage collection...")
    gc.collect()
    gc.collect()

    print("[2] CUDA context reset...")
    if torch.cuda.is_available():
        # Clear all cached memory
        torch.cuda.empty_cache()

        # Clear IPC handles
        try:
            torch.cuda.ipc_collect()
            print("    [OK] IPC handles cleared")
        except Exception:
            print("    [WARN] IPC collect not available")

        # Synchronize
        torch.cuda.synchronize()
        print("    [OK] CUDA synchronized")

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        print("    [OK] Memory stats reset")

        # Log memory state
        free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        print(f"\n[3] After cleanup: {free_mem:.2f}GB free GPU memory")

    # Final GC
    gc.collect()

    # Stabilization delay
    print("[4] Waiting 3 seconds for CUDA context to stabilize...")
    time.sleep(3)

    print("\n[OK] Cleanup complete")


def test_quiz_loading():
    """Test Quiz model loading after cleanup."""
    print("\n" + "="*80)
    print("LOADING QUIZ MODEL")
    print("="*80)

    try:
        from quiz_generator_v2 import QuizGeneratorV2

        print("\n[1] Initializing Quiz Generator...")
        quiz = QuizGeneratorV2()

        print("[2] Loading model (this is where segfault occurred)...")
        start = time.time()

        quiz.load_model()

        elapsed = time.time() - start

        print(f"\n✓ Model loaded successfully in {elapsed:.2f}s!")

        # Check memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ GPU Memory: {allocated:.2f}GB / {total:.2f}GB")

        # Quick generation test
        print("\n[3] Testing generation...")
        test_prompt = "What is psychology?"

        result = quiz.generate_mcq_questions(
            context=test_prompt,
            num_questions=1
        )

        if result and len(result) > 0:
            print(f"✓ Generated {len(result)} question(s)")
            print(f"\nSample question:")
            q = result[0]
            print(f"Q: {q.get('question', 'N/A')}")
        else:
            print("⚠ No questions generated")

        # Cleanup
        print("\n[4] Cleaning up Quiz model...")
        quiz.unload_model()
        print("✓ Quiz model unloaded")

        return True

    except Exception as e:
        print(f"\n✗ Quiz loading FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the complete test."""
    print("="*80)
    print("QUIZ MODEL LOADING TEST - AFTER AGGRESSIVE CLEANUP")
    print("="*80)
    print("\nThis test simulates the pipeline scenario:")
    print("1. Load/unload a model (like Notes)")
    print("2. Aggressive cleanup")
    print("3. Load Quiz model")
    print("\nObjective: Verify no segmentation fault occurs")

    # Step 1: Simulate previous model
    simulate_previous_model()

    # Step 2: Aggressive cleanup
    aggressive_cleanup()

    # Step 3: Load Quiz
    success = test_quiz_loading()

    # Final result
    print("\n" + "="*80)
    if success:
        print("✓ TEST PASSED - Quiz loaded successfully after cleanup!")
        print("="*80)
        print("\nConclusion:")
        print("- Aggressive cleanup strategy works")
        print("- Quiz model can load after previous models")
        print("- Ready for full pipeline test")
    else:
        print("✗ TEST FAILED - Quiz still has issues")
        print("="*80)
        print("\nNext steps:")
        print("1. Check Windows paging file (should be 16-32GB)")
        print("2. Verify CUDA drivers are up to date")
        print("3. Try INT8 quantization instead of 4-bit")

    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
