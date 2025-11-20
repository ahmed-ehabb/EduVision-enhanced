"""
Test Open-Ended Questions WITH Sample Answers
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from quiz_generator_v2 import QuizGeneratorV2

def test_with_answers():
    """Test open-ended questions with sample answers."""
    
    print("="*80)
    print("OPEN-ENDED QUESTIONS WITH SAMPLE ANSWERS TEST")
    print("="*80)
    
    # Sample context
    context = """
    Classical conditioning was discovered by Ivan Pavlov. In his experiments, 
    dogs learned to salivate at the sound of a bell paired with food. The neutral 
    stimulus (bell) became a conditioned stimulus that elicited salivation.
    
    Operant conditioning, studied by B.F. Skinner, involves learning through 
    consequences. Behaviors followed by positive outcomes are reinforced.
    """
    
    # Initialize
    print("\n[1] Loading Quiz Generator...")
    quiz_gen = QuizGeneratorV2()
    
    try:
        quiz_gen.load_model()
        print("    [OK] Model loaded\n")
        
        # Generate questions WITHOUT answers
        print("[2] Generating questions WITHOUT answers...")
        questions_only = quiz_gen.generate_open_ended_questions(
            context=context,
            num_questions=2,
            question_type="short_answer",
            include_answers=False,
            max_new_tokens=300
        )
        
        print("\n--- QUESTIONS ONLY ---")
        print(questions_only)
        print("\n")
        
        # Generate questions WITH answers
        print("[3] Generating questions WITH sample answers...")
        questions_with_answers = quiz_gen.generate_open_ended_questions(
            context=context,
            num_questions=2,
            question_type="short_answer",
            include_answers=True,
            max_new_tokens=600
        )
        
        print("\n--- QUESTIONS WITH SAMPLE ANSWERS ---")
        print(questions_with_answers)
        print("\n")
        
        # Cleanup
        print("[4] Cleaning up...")
        quiz_gen.unload_model()
        print("    [OK] Model unloaded\n")
        
        print("="*80)
        print("[SUCCESS] Test completed!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_answers()
    sys.exit(0 if success else 1)
