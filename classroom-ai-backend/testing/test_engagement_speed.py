"""
Quick Engagement Speed Test
===========================

Tests engagement analyzer speed with optimization.
"""

import sys
import time
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from engagement_analyzer_v2 import EngagementAnalyzerV2

def test_engagement_speed():
    """Test engagement speed with lecture_1.mp3"""

    print("="*80)
    print("Engagement Analysis Speed Test")
    print("="*80)

    # Audio file
    audio_path = Path(__file__).parent / "lecture_1.mp3"

    if not audio_path.exists():
        print(f"[FAIL] Audio not found: {audio_path}")
        return False

    print(f"\n[INFO] Audio: {audio_path.name}")

    # Dummy transcript segments (50 segments like in real test)
    dummy_segments = ["segment text"] * 50

    # Create analyzer
    print("\n[INFO] Creating analyzer...")
    analyzer = EngagementAnalyzerV2()

    # Analyze
    print("\n[INFO] Running engagement analysis...")
    print("-"*80)

    start = time.time()
    result = analyzer.analyze_audio(str(audio_path), dummy_segments)
    elapsed = time.time() - start

    print("-"*80)
    print(f"\n[OK] Analysis complete!")
    print(f"[INFO] Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"[INFO] Score: {result.get('engagement_score', 0):.2f}%")
    print(f"[INFO] Segments: {len(result.get('segment_scores', []))}")

    # Expected performance
    if elapsed < 120:  # Target: < 2 minutes
        print(f"[OK] Performance EXCELLENT: {elapsed:.1f}s < 120s")
    elif elapsed < 300:  # < 5 minutes
        print(f"[OK] Performance GOOD: {elapsed:.1f}s < 300s")
    else:
        print(f"[WARNING] Performance SLOW: {elapsed:.1f}s > 300s")

    print("\n" + "="*80)
    print("Test Complete")
    print("="*80)

    return True


if __name__ == "__main__":
    try:
        success = test_engagement_speed()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
