"""
Test Open-Ended Question Generation
====================================

Tests the new generate_open_ended_questions() method.
"""

import sys
import time
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from quiz_generator_v2 import QuizGeneratorV2


def test_open_ended_questions():
    """Test all three types of open-ended questions."""

    print("="*80)
    print("OPEN-ENDED QUESTION GENERATION TEST")
    print("="*80)

    # Sample context from psychology
    context = """
    Psychology is the scientific study of mind and behavior. The discipline encompasses
    the scientific study of mental processes and behavior and the application of this
    knowledge to understand and address psychological problems.

    Classical conditioning is a learning process first described by Ivan Pavlov. In classical
    conditioning, a neutral stimulus is paired with a naturally occurring stimulus until the
    neutral stimulus alone produces the same response. For example, Pavlov's dogs learned
    to salivate at the sound of a bell because it had been paired with food.

    Operant conditioning, developed by B.F. Skinner, focuses on how consequences shape behavior.
    Reinforcement increases the likelihood of a behavior, while punishment decreases it.
    Positive reinforcement adds something desirable, while negative reinforcement removes
    something undesirable.
    """

    # Load Quiz Generator
    print("\n[1] Loading Quiz Generator...")
    quiz_gen = QuizGeneratorV2()

    start = time.time()
    quiz_gen.load_model()
    load_time = time.time() - start

    print(f"    [OK] Model loaded in {load_time:.1f}s")

    # Test each question type
    question_types = [
        ("short_answer", 3),
        ("essay", 2),
        ("discussion", 2)
    ]

    for q_type, num_questions in question_types:
        print(f"\n[2] Generating {num_questions} {q_type.replace('_', ' ')} questions...")
        print("-"*80)

        start = time.time()
        result = quiz_gen.generate_open_ended_questions(
            context=context,
            num_questions=num_questions,
            question_type=q_type
        )
        elapsed = time.time() - start

        print(f"\n[Generated in {elapsed:.1f}s]")
        print(result)
        print("-"*80)

    # Cleanup
    print("\n[3] Cleaning up...")
    quiz_gen.unload_model()
    print("    [OK] Model unloaded")

    print("\n" + "="*80)
    print("TEST COMPLETE - All question types generated successfully!")
    print("="*80)

    return True


if __name__ == "__main__":
    try:
        success = test_open_ended_questions()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
