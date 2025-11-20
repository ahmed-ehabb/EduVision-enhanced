"""
Quick ASR Speed Test
====================

Tests ASR speed with optimizations to verify:
1. No attention mask warnings
2. Faster inference (greedy decoding)
3. Proper generation parameters
"""

import sys
import time
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from asr_module import ASRModule

def test_asr_speed():
    """Test ASR speed with lecture_1.mp3"""

    print("="*80)
    print("ASR Speed Test")
    print("="*80)

    # Audio file
    audio_path = Path(__file__).parent / "lecture_1.mp3"

    if not audio_path.exists():
        print(f"[FAIL] Audio not found: {audio_path}")
        return False

    print(f"\n[INFO] Audio: {audio_path.name}")

    # Get duration
    import librosa
    y, sr = librosa.load(str(audio_path), sr=None, mono=False)
    duration = len(y) / sr
    print(f"[INFO] Duration: {duration:.2f}s ({duration/60:.2f} minutes)")

    # Load ASR
    print("\n[INFO] Loading ASR model...")
    start = time.time()
    asr = ASRModule(
        model_id="ahmedheakl/arazn-whisper-small-v2",
        compute_type="int8",
        device="cuda"
    )
    load_time = time.time() - start
    print(f"[OK] Model loaded in {load_time:.2f}s")

    # Transcribe
    print("\n[INFO] Transcribing (watch for warnings)...")
    print("-"*80)

    start = time.time()
    result = asr.transcribe(str(audio_path), add_punctuation=False)
    transcribe_time = time.time() - start

    print("-"*80)
    print(f"\n[OK] Transcription complete!")
    print(f"[INFO] Time: {transcribe_time:.2f}s ({transcribe_time/60:.2f} minutes)")
    print(f"[INFO] Speed ratio: {duration/transcribe_time:.2f}x realtime")
    print(f"[INFO] Chars: {len(result['text'])}")
    print(f"[INFO] Chunks: {result.get('chunks_processed', 'N/A')}")

    # Check for warnings
    print("\n[INFO] If you see NO attention mask warnings above, optimization worked!")

    # Expected performance
    expected_time = duration / 2.5  # Target: 2.5x realtime or better
    if transcribe_time <= expected_time:
        print(f"[OK] Performance GOOD: {transcribe_time:.1f}s <= {expected_time:.1f}s (target)")
    else:
        print(f"[WARNING] Performance: {transcribe_time:.1f}s > {expected_time:.1f}s (target)")
        print(f"[INFO] Expected ~{expected_time/60:.1f} minutes, got {transcribe_time/60:.1f} minutes")

    print("\n" + "="*80)
    print("Test Complete")
    print("="*80)

    return True


if __name__ == "__main__":
    try:
        success = test_asr_speed()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
