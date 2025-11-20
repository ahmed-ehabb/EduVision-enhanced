"""
Test Report Generator
====================

Tests the report generation system with sample data.
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from report_generator import ReportGenerator


def test_report_generator():
    """Test report generator with comprehensive sample data."""

    print("="*80)
    print("Testing Report Generator")
    print("="*80)

    # Sample comprehensive results
    sample_results = {
        'total_time': 32.5,
        'transcript_length': 6356,
        'engagement': {
            'engagement_score': 65.38,
            'detailed_scores': {
                'active_learning': 72.5,
                'clarity': 68.2,
                'interaction': 55.3,
                'enthusiasm': 65.9
            }
        },
        'content_alignment': {
            'coverage_percentage': 11.54,
            'num_covered_topics': 3,
            'num_total_topics': 26,
            'covered_topics': [
                'Introduction to Psychology',
                'Learning Theories',
                'Behavioral Psychology'
            ]
        },
        'notes': {
            'bullet_points': [
                'Psychology is the scientific study of mind and behavior.',
                'Behavioral psychology focuses on observable behaviors rather than internal mental states.',
                'Learning can occur through reinforcement and punishment mechanisms.',
                'Cognitive psychology examines mental processes such as memory, perception, and problem-solving.',
                'The scientific method is fundamental to psychological research.'
            ]
        },
        'quiz': {
            'questions': [
                {
                    'question': 'What is the primary focus of behavioral psychology?',
                    'options': [
                        'Mental processes and thoughts',
                        'Observable behaviors and actions',
                        'Emotional states and feelings',
                        'Unconscious mind and dreams'
                    ],
                    'correct_answer': 'B',
                    'explanation': 'Behavioral psychology focuses on observable behaviors rather than internal mental states, as demonstrated by the work of B.F. Skinner and John Watson.'
                },
                {
                    'question': 'Who is considered the father of behaviorism?',
                    'options': [
                        'Sigmund Freud',
                        'B.F. Skinner',
                        'John Watson',
                        'Carl Rogers'
                    ],
                    'correct_answer': 'C',
                    'explanation': 'John Watson is considered the father of behaviorism, though B.F. Skinner made significant contributions to the field later.'
                },
                {
                    'question': 'What is reinforcement in learning theory?',
                    'options': [
                        'A consequence that decreases behavior',
                        'A consequence that increases behavior',
                        'A neutral stimulus',
                        'An unconscious process'
                    ],
                    'correct_answer': 'B',
                    'explanation': 'Reinforcement is any consequence that increases the likelihood of a behavior being repeated in the future.'
                },
                {
                    'question': 'Which area of psychology studies memory and perception?',
                    'options': [
                        'Behavioral psychology',
                        'Cognitive psychology',
                        'Psychoanalysis',
                        'Humanistic psychology'
                    ],
                    'correct_answer': 'B',
                    'explanation': 'Cognitive psychology focuses on mental processes including memory, perception, thinking, and problem-solving.'
                },
                {
                    'question': 'What is the scientific method in psychology?',
                    'options': [
                        'A way to interpret dreams',
                        'A systematic approach to research',
                        'A therapy technique',
                        'A personality test'
                    ],
                    'correct_answer': 'B',
                    'explanation': 'The scientific method is a systematic approach to research that involves observation, hypothesis formation, experimentation, and analysis.'
                }
            ]
        },
        'translation': {
            'arabic_text': 'علم النفس هو الدراسة العلمية للعقل والسلوك...',
            'success': True
        }
    }

    # Create report generator
    print("\n[INFO] Initializing report generator...")
    generator = ReportGenerator(output_dir="./reports")

    # Generate PDF report
    print("\n[INFO] Generating PDF report...")
    try:
        pdf_path = generator.generate_pdf_report(
            sample_results,
            "Introduction to Psychology - Lecture 1",
            "test_report.pdf"
        )
        print(f"[OK] PDF report generated: {pdf_path}")
    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Generate HTML report
    print("\n[INFO] Generating HTML report...")
    try:
        html_path = generator.generate_html_report(
            sample_results,
            "Introduction to Psychology - Lecture 1",
            "test_report.html"
        )
        print(f"[OK] HTML report generated: {html_path}")
    except Exception as e:
        print(f"[ERROR] HTML generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Generate both formats at once
    print("\n[INFO] Generating both formats...")
    try:
        paths = generator.generate_reports(
            sample_results,
            "Introduction to Psychology - Lecture 1",
            formats=['pdf', 'html']
        )

        print("\n[OK] All reports generated:")
        for fmt, path in paths.items():
            print(f"  {fmt.upper()}: {path}")
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("Report Generator Test Complete")
    print("="*80)

    return True


if __name__ == "__main__":
    try:
        success = test_report_generator()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
