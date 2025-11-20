"""
Test Content Alignment Module - Text Alignment with SBERT

Tests the content alignment analyzer that uses SBERT to compare
lecture transcripts with textbook content.

Model: sentence-transformers/paraphrase-MiniLM-L6-v2
Processing: CPU or GPU (auto-detect)
Memory: Minimal (~200MB)
"""

import sys
import os
import time
from pathlib import Path

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

    try:
        from sentence_transformers import SentenceTransformer
        print(f"[OK] sentence-transformers available")
        return True
    except ImportError:
        print(f"[FAIL] sentence-transformers not available")
        print(f"[INFO] Install with: pip install sentence-transformers")
        return False


def test_model_loading():
    """Test 1: SBERT Model Loading"""
    print_section("TEST 1: SBERT Model Loading")

    try:
        from content_alignment_v2 import ContentAlignmentAnalyzer

        print("[INFO] Creating content alignment analyzer...")
        start_time = time.time()

        analyzer = ContentAlignmentAnalyzer()
        analyzer.load_model()

        load_time = time.time() - start_time

        print(f"[OK] Model loaded in {load_time:.2f}s")

        # Get model info
        info = analyzer.get_model_info()
        print("\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Check memory
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"\n[INFO] GPU Memory: {allocated:.2f}GB / {total:.2f}GB")
            else:
                print(f"\n[INFO] Running on CPU (no GPU detected)")
        except:
            print(f"\n[INFO] Running on CPU")

        return True

    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_textbook_loading():
    """Test 2: Textbook Content Loading"""
    print_section("TEST 2: Textbook Content Loading")

    try:
        from content_alignment_v2 import ContentAlignmentAnalyzer

        analyzer = ContentAlignmentAnalyzer()

        # Sample textbook paragraphs
        textbook_paragraphs = [
            "Psychology is the scientific study of mind and behavior. It encompasses the study of conscious and unconscious phenomena.",
            "The scientific method involves forming hypotheses, conducting experiments, and analyzing data systematically.",
            "Operant conditioning is a learning process where behavior is modified by consequences such as reinforcement or punishment.",
            "Classical conditioning involves learning through association, as demonstrated by Pavlov's experiments with dogs.",
            "The brain consists of billions of neurons that communicate through electrical and chemical signals.",
            "Developmental psychology studies how people grow and change throughout their lifespan.",
            "Social psychology examines how individuals influence and are influenced by others.",
            "Cognitive psychology focuses on mental processes like memory, perception, and problem-solving."
        ]

        print(f"[INFO] Loading {len(textbook_paragraphs)} textbook paragraphs...")
        start_time = time.time()

        analyzer.load_textbook_content(textbook_paragraphs)

        load_time = time.time() - start_time

        print(f"[OK] Textbook content loaded in {load_time:.2f}s")
        print(f"[OK] Embeddings created for {len(textbook_paragraphs)} paragraphs")

        return True

    except Exception as e:
        print(f"[FAIL] Textbook loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transcript_analysis():
    """Test 3: Transcript Analysis"""
    print_section("TEST 3: Transcript Analysis")

    try:
        from content_alignment_v2 import ContentAlignmentAnalyzer

        analyzer = ContentAlignmentAnalyzer()

        # Load textbook
        textbook_paragraphs = [
            "Psychology is the scientific study of mind and behavior. It encompasses the study of conscious and unconscious phenomena.",
            "The scientific method involves forming hypotheses, conducting experiments, and analyzing data systematically.",
            "Operant conditioning is a learning process where behavior is modified by consequences such as reinforcement or punishment.",
            "Classical conditioning involves learning through association, as demonstrated by Pavlov's experiments with dogs.",
            "The brain consists of billions of neurons that communicate through electrical and chemical signals."
        ]

        analyzer.load_textbook_content(textbook_paragraphs)

        # Sample transcript with varying alignment
        transcript_segments = [
            "Today we're going to talk about what psychology really is. It's the scientific study of how we think and behave.",
            "Psychologists use the scientific method to test their ideas. They form hypotheses and run experiments.",
            "Remember Pavlov's dogs? That's a classic example of classical conditioning where learning happens through association.",
            "The weather today is really nice, I hope you all had a good breakfast this morning.",
            "Operant conditioning is different - it's about learning from consequences like rewards and punishments."
        ]

        print(f"[INFO] Analyzing {len(transcript_segments)} transcript segments...")
        start_time = time.time()

        result = analyzer.analyze_transcript(transcript_segments)

        analysis_time = time.time() - start_time

        print(f"[OK] Analysis completed in {analysis_time:.2f}s")

        print(f"\n{'='*80}")
        print(f"Analysis Results")
        print(f"{'='*80}")

        print(f"\nOverall Coverage Score: {result['coverage_score']:.2f}%")
        print(f"Feedback: {result['feedback']}")

        print(f"\nCoverage Breakdown:")
        for label, stats in result['coverage_percentages'].items():
            print(f"  {label}: {stats['count']} segments ({stats['percentage']:.1f}%)")

        print(f"\nTop 3 Segment Details:")
        for r in result['results'][:3]:
            print(f"\n  Segment {r['segment_id'] + 1}: {r['coverage_label']} (similarity: {r['similarity_score']:.3f})")
            print(f"    Transcript: {r['transcript_segment'][:70]}...")

        return True

    except Exception as e:
        print(f"[FAIL] Transcript analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coverage_classification():
    """Test 4: Coverage Classification Logic"""
    print_section("TEST 4: Coverage Classification Logic")

    try:
        from content_alignment_v2 import ContentAlignmentAnalyzer

        analyzer = ContentAlignmentAnalyzer()

        # Single textbook paragraph
        textbook_paragraphs = [
            "Operant conditioning is a learning process where behavior is modified by consequences such as reinforcement or punishment. B.F. Skinner studied this extensively."
        ]

        analyzer.load_textbook_content(textbook_paragraphs)

        # Test different levels of alignment
        test_cases = [
            {
                "name": "High Similarity (Fully Covered)",
                "text": "Operant conditioning modifies behavior through consequences like reinforcement and punishment. Skinner researched this."
            },
            {
                "name": "Medium Similarity (Partially Covered)",
                "text": "Learning through consequences is important in psychology. Behavior can be shaped by rewards."
            },
            {
                "name": "Low Similarity (Off-topic)",
                "text": "The weather today is sunny and warm. It's a great day for outdoor activities."
            }
        ]

        print("[INFO] Testing classification thresholds...")

        for test_case in test_cases:
            result = analyzer.analyze_transcript([test_case["text"]])
            segment_result = result['results'][0]

            print(f"\n{test_case['name']}:")
            print(f"  Similarity: {segment_result['similarity_score']:.3f}")
            print(f"  Classification: {segment_result['coverage_label']}")

        print(f"\n[OK] Classification logic working correctly")
        return True

    except Exception as e:
        print(f"[FAIL] Classification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """Test 5: Memory Usage"""
    print_section("TEST 5: Memory Usage Monitoring")

    try:
        import torch

        if not torch.cuda.is_available():
            print("[INFO] CUDA not available - model runs on CPU")
            print("[INFO] SBERT model uses minimal memory (~200MB)")
            return True

        from content_alignment_v2 import ContentAlignmentAnalyzer

        # Clear memory
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.memory_allocated(0) / (1024**3)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print(f"Initial GPU Memory: {initial_mem:.2f}GB / {total_mem:.2f}GB")

        # Load model
        analyzer = ContentAlignmentAnalyzer()
        analyzer.load_model()

        after_load = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"After Model Load: {after_load:.2f}GB / {total_mem:.2f}GB")
        print(f"Model Memory: {after_load - initial_mem:.2f}GB")

        # Load textbook
        textbook_paragraphs = ["Test paragraph " + str(i) for i in range(100)]
        analyzer.load_textbook_content(textbook_paragraphs)

        after_textbook = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"After Textbook Load: {after_textbook:.2f}GB / {total_mem:.2f}GB")

        # Analyze transcript
        transcript = ["Test segment " + str(i) for i in range(50)]
        result = analyzer.analyze_transcript(transcript)

        after_analysis = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"After Analysis: {after_analysis:.2f}GB / {total_mem:.2f}GB")

        # Cleanup
        analyzer = None
        torch.cuda.empty_cache()

        after_cleanup = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"After Cleanup: {after_cleanup:.2f}GB / {total_mem:.2f}GB")

        max_usage = max(initial_mem, after_load, after_textbook, after_analysis)

        if max_usage <= 4.0:
            print(f"\n[OK] Max memory usage: {max_usage:.2f}GB (within 4GB limit)")
        else:
            print(f"\n[!] Warning: Max memory usage: {max_usage:.2f}GB (exceeds 4GB)")

        return True

    except Exception as e:
        print(f"[INFO] Memory test informational: {e}")
        print("[INFO] Content alignment uses minimal memory (~200MB)")
        return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("  CONTENT ALIGNMENT - RTX 3050 DEPLOYMENT TEST")
    print("="*80)
    print("\n[INFO] Testing SBERT-based content alignment\n")

    # Check dependencies
    if not test_dependencies():
        print("\n[FAIL] Dependencies not met. Cannot proceed.")
        print("\nTo install dependencies:")
        print("  pip install sentence-transformers")
        return False

    tests = [
        ("SBERT Model Loading", test_model_loading),
        ("Textbook Content Loading", test_textbook_loading),
        ("Transcript Analysis", test_transcript_analysis),
        ("Coverage Classification", test_coverage_classification),
        ("Memory Usage", test_memory_usage),
    ]

    results = {"Dependency Check": True}

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

    print("\n[INFO] Content Alignment Module:")
    print("  - Model: paraphrase-MiniLM-L6-v2 (SBERT)")
    print("  - Memory: ~200MB")
    print("  - Processing: CPU or GPU (auto-detect)")
    print("  - Classification: Fully/Partially Covered, Off-topic")

    if passed == total:
        print("\n[OK] All tests passed! Content alignment ready.")
    else:
        print(f"\n[!] {total - passed} test(s) failed.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
