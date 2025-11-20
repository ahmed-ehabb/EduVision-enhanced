"""
Test Quiz Question Parsing
===========================

Tests that generated MCQ questions are parsed into structured format.
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from quiz_generator_v2 import QuizGeneratorV2


def test_quiz_parsing():
    """Test Quiz parsing returns structured questions."""

    print("="*80)
    print("QUIZ PARSING TEST")
    print("="*80)

    # Sample context
    context = """
    Classical conditioning is a learning process discovered by Ivan Pavlov.
    In classical conditioning, a neutral stimulus is repeatedly paired with
    an unconditioned stimulus until the neutral stimulus alone produces a response.
    """

    # Load Quiz Generator
    print("\n[1] Loading Quiz Generator...")
    quiz_gen = QuizGeneratorV2()
    quiz_gen.load_model()
    print("    [OK] Model loaded")

    # Generate questions
    print("\n[2] Generating 3 MCQ questions...")
    questions = quiz_gen.generate_mcq_questions(
        context=context,
        num_questions=3,
        max_new_tokens=500
    )

    print(f"    [OK] Generated {len(questions)} questions")

    # Verify structure
    print("\n[3] Verifying question structure...")

    for i, q in enumerate(questions, 1):
        print(f"\n    Question {i}:")
        print(f"      Type: {type(q)}")

        # Check if it's a dict
        if isinstance(q, dict):
            print(f"      Has 'question' key: {'question' in q}")
            print(f"      Has 'options' key: {'options' in q}")
            print(f"      Has 'correct_answer' key: {'correct_answer' in q}")

            if 'question' in q:
                print(f"      Question text: {q['question'][:60]}...")

            if 'options' in q and isinstance(q['options'], dict):
                print(f"      Options count: {len(q['options'])}")
                for opt_key in ['a', 'b', 'c', 'd']:
                    if opt_key in q['options']:
                        print(f"        {opt_key}) {q['options'][opt_key][:40]}...")

            if 'correct_answer' in q:
                print(f"      Correct answer: {q['correct_answer']}")
        else:
            print(f"      [ERROR] Not a dictionary! Type: {type(q)}")

    # Cleanup
    print("\n[4] Cleaning up...")
    quiz_gen.unload_model()
    print("    [OK] Model unloaded")

    # Final check
    print("\n" + "="*80)
    if all(isinstance(q, dict) and 'question' in q for q in questions):
        print("[SUCCESS] All questions are properly structured!")
        print("="*80)
        return True
    else:
        print("[FAIL] Some questions are not properly structured")
        print("="*80)
        return False


if __name__ == "__main__":
    try:
        success = test_quiz_parsing()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
