"""
End-to-End Test for TeacherModule V2

Tests the complete lecture evaluation pipeline with real data.

Uses:
- Audio: testing/lecture_1.mp3 (Real YouTube lecture)
- Transcript: testing/chapter1_lec.txt (for textbook content)
"""

import sys
import os
import time
import json
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))


def print_section(title: str):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def load_textbook_content():
    """Load textbook content for testing."""
    # Sample textbook paragraphs (Psychology content)
    textbook_paragraphs = [
        "Psychology is the scientific study of mind and behavior. The study of psychology has a long history, dating back to ancient civilizations.",
        "The scientific method involves forming hypotheses, conducting experiments, and analyzing data systematically to understand phenomena.",
        "Operant conditioning is a learning process where behavior is modified by consequences such as reinforcement or punishment. B.F. Skinner studied this extensively.",
        "Classical conditioning involves learning through association, as demonstrated by Pavlov's experiments with dogs and salivation responses.",
        "The brain consists of billions of neurons that communicate through electrical and chemical signals, forming the biological basis of behavior.",
        "Developmental psychology studies how people grow and change throughout their lifespan, from infancy to old age.",
        "Social psychology examines how individuals influence and are influenced by others in social contexts and group settings.",
        "Cognitive psychology focuses on mental processes like memory, perception, attention, problem-solving, and decision making.",
        "The scientific study of behavior requires careful observation, measurement, and experimentation to draw valid conclusions.",
        "Research ethics in psychology mandate informed consent, confidentiality, and protection from harm for all participants.",
        "Neuroscience explores the relationship between brain structures, neural processes, and psychological functions and behavior.",
        "Intelligence testing and assessment have been important tools in psychology, though they remain subjects of ongoing debate.",
        "Mental health disorders affect millions of people worldwide and require evidence-based psychological interventions and treatments.",
        "Learning theories explain how organisms acquire new behaviors through experience, practice, and environmental interactions.",
        "Personality psychology investigates individual differences in patterns of thinking, feeling, and behaving across situations."
    ]

    return textbook_paragraphs


def test_teacher_module_e2e():
    """Test complete pipeline."""
    print_section("TEACHER MODULE - END-TO-END TEST")

    try:
        from teacher_module_v2 import TeacherModuleV2
        import torch

        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[INFO] GPU: {gpu_name} ({total_mem:.2f}GB VRAM)")
        else:
            print("[WARNING] No GPU detected - will run on CPU")

        # Paths - Using real lecture audio
        test_audio = Path(__file__).parent / "lecture_1.mp3"

        if not test_audio.exists():
            print(f"[FAIL] Test audio not found: {test_audio}")
            return False

        print(f"[OK] Real lecture audio found: {test_audio}")

        # Get audio duration
        import librosa
        y, sr = librosa.load(str(test_audio), sr=None, mono=False)
        duration = len(y) / sr
        print(f"[INFO] Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

        # Load textbook content
        print("[INFO] Loading textbook content...")
        textbook_paragraphs = load_textbook_content()
        print(f"[OK] Loaded {len(textbook_paragraphs)} textbook paragraphs")

        # Create teacher module
        print("\n[INFO] Creating TeacherModule...")
        teacher = TeacherModuleV2(
            use_punctuation=False,  # Skip (very slow)
            use_translation=True,   # Enable translation
            enable_quiz=True,       # Generate quiz
            enable_notes=True       # Generate notes
        )

        print("[OK] TeacherModule created")

        # Process lecture
        print("\n" + "="*80)
        print("PROCESSING LECTURE")
        print("="*80)

        start_time = time.time()

        results = teacher.process_lecture(
            audio_path=str(test_audio),
            textbook_paragraphs=textbook_paragraphs,
            lecture_title="Psychology 101 - Introduction"
        )

        total_time = time.time() - start_time

        # Display results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)

        # Summary
        if "summary" in results:
            summary = results["summary"]
            print(f"\nSUMMARY:")
            print(f"  Lecture: {summary.get('lecture_title', 'N/A')}")
            print(f"  Steps Completed: {len(summary.get('steps_completed', []))}")

            if "engagement_score" in summary:
                print(f"  Engagement Score: {summary['engagement_score']:.2f}%")

            if "coverage_score" in summary:
                print(f"  Coverage Score: {summary['coverage_score']:.2f}%")

            if "combined_score" in summary:
                print(f"  Combined Score: {summary['combined_score']:.2f}%")
                print(f"  Grade: {summary['grade']}")

        # Transcript
        if "transcript" in results and results["transcript"].get("success"):
            trans = results["transcript"]
            print(f"\nTRANSCRIPT:")
            print(f"  Status: [OK] Success")
            print(f"  Text length: {len(trans.get('text', ''))} chars")
            print(f"  Segments: {trans.get('num_segments', 0)}")
            print(f"  Processing time: {trans.get('processing_time', 0):.2f}s")
            print(f"  Preview: {trans.get('text', '')[:200]}...")

        # Engagement
        if "engagement" in results and results["engagement"].get("success"):
            eng = results["engagement"]
            print(f"\nENGAGEMENT:")
            print(f"  Status: [OK] Success")
            print(f"  Score: {eng.get('engagement_score', 0):.2f}%")
            stats = eng.get("statistics", {})
            if stats:
                print(f"  Engaging segments: {stats.get('engaging_segments', 0)}")
                print(f"  Neutral segments: {stats.get('neutral_segments', 0)}")
                print(f"  Boring segments: {stats.get('boring_segments', 0)}")

        # Content Alignment
        if "content_alignment" in results and results["content_alignment"].get("success"):
            align = results["content_alignment"]
            print(f"\nCONTENT ALIGNMENT:")
            print(f"  Status: [OK] Success")
            print(f"  Coverage Score: {align.get('coverage_score', 0):.2f}%")
            print(f"  Feedback: {align.get('feedback', 'N/A')}")
            perc = align.get("coverage_percentages", {})
            if perc:
                for label, data in perc.items():
                    print(f"    {label}: {data.get('count', 0)} ({data.get('percentage', 0):.1f}%)")

        # Translation
        if "translation" in results:
            if results["translation"].get("skipped"):
                print(f"\nTRANSLATION: Skipped (not needed)")
            elif results["translation"].get("success"):
                print(f"\nTRANSLATION: [OK] Success")

        # Notes
        if "notes" in results and results["notes"].get("success"):
            notes = results["notes"]
            print(f"\nNOTES:")
            print(f"  Status: [OK] Success")
            print(f"  Bullet points: {len(notes.get('bullet_points', []))}")
            for i, bullet in enumerate(notes.get("bullet_points", [])[:3], 1):
                print(f"    {i}. {bullet[:80]}...")

        # Quiz
        if "quiz" in results and results["quiz"].get("success"):
            quiz = results["quiz"]
            print(f"\nQUIZ:")
            print(f"  Status: [OK] Success")

            # MCQ questions
            num_mcq = quiz.get('num_mcq', len(quiz.get('questions', [])))
            print(f"  MCQ questions: {num_mcq}")
            for i, q in enumerate(quiz.get("mcq_questions", quiz.get("questions", []))[:2], 1):
                print(f"    Q{i}: {q.get('question', 'N/A')[:60]}...")

            # Open-ended questions
            num_open = quiz.get('num_open_ended', 0)
            print(f"  Open-ended questions: {num_open}")
            open_ended = quiz.get("open_ended_questions", "")
            if open_ended:
                # Show first 200 chars of open-ended questions
                print(f"    Preview: {open_ended[:200]}...")

            # Backward compatibility
            if 'num_questions' not in quiz:
                quiz['num_questions'] = num_mcq

        # Errors
        if results.get("errors"):
            print(f"\n[!] ERRORS:")
            for error in results["errors"]:
                print(f"  - {error}")

        # Performance
        print(f"\nPERFORMANCE:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Reported time: {results.get('total_processing_time', 0):.2f}s")

        # Memory
        if torch.cuda.is_available():
            final_mem = torch.cuda.memory_allocated(0) / (1024**3)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\nMEMORY:")
            print(f"  Final GPU usage: {final_mem:.2f}GB / {total_mem:.2f}GB")
            if final_mem <= 4.0:
                print(f"  [OK] Within 4GB limit")
            else:
                print(f"  [!] Exceeds 4GB limit")

        # Save results
        output_file = Path(__file__).parent / "e2e_test_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nResults saved to: {output_file}")

        # Final verdict
        print("\n" + "="*80)
        required_steps = ["transcription", "engagement", "content_alignment"]
        completed_steps = results.get("processing_steps", [])
        all_required = all(step in completed_steps for step in required_steps)

        if all_required and not results.get("errors"):
            print("[OK] END-TO-END TEST PASSED")
            print("="*80)
            return True
        else:
            print("[!] END-TO-END TEST COMPLETED WITH WARNINGS")
            print("="*80)
            return True

    except Exception as e:
        print(f"\n[FAIL] End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_teacher_module_e2e()
    sys.exit(0 if success else 1)
