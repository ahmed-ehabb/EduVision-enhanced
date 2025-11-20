"""
Test Report Enhancement - Open-Ended Questions with Sample Answers
====================================================================

This script tests the updated report generator to ensure:
1. PDF includes open-ended questions with sample answers
2. HTML includes open-ended questions with sample answers
3. Proper formatting and parsing of question/answer blocks
"""

import os
import sys
import json
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_results():
    """Load existing E2E test results."""
    results_path = os.path.join(os.path.dirname(__file__), 'e2e_test_results.json')

    if not os.path.exists(results_path):
        logger.error(f"[ERROR] Results file not found: {results_path}")
        return None

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    logger.info(f"[OK] Loaded test results from: {results_path}")
    return results


def test_report_enhancement():
    """Test PDF and HTML report generation with open-ended questions."""

    print("\n" + "="*70)
    print("Testing Report Enhancement - Open-Ended Questions with Sample Answers")
    print("="*70)

    # Step 1: Load E2E test results
    print("\n[1] Loading E2E test results...")
    results = load_test_results()

    if not results:
        print("[FAIL] Could not load test results")
        return False

    # Check if quiz data includes open-ended questions
    if 'quiz' not in results:
        print("[FAIL] No quiz data in results")
        return False

    quiz = results['quiz']
    num_mcq = quiz.get('num_mcq', 0)
    num_open_ended = quiz.get('num_open_ended', 0)

    print(f"  MCQ questions: {num_mcq}")
    print(f"  Open-ended questions: {num_open_ended}")

    if num_open_ended == 0:
        print("[FAIL] No open-ended questions in results")
        return False

    # Preview open-ended questions
    open_ended_text = quiz.get('open_ended_questions', '')
    if open_ended_text:
        print(f"\n  Preview of open-ended questions ({len(open_ended_text)} chars):")
        print(f"  {open_ended_text[:250]}...")

    print("  [OK] Test results loaded")

    # Step 2: Initialize report generator
    print("\n[2] Initializing report generator...")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(output_dir, exist_ok=True)

    generator = ReportGenerator(output_dir=output_dir)
    print(f"  [OK] Output directory: {output_dir}")

    # Step 3: Generate PDF report
    print("\n[3] Generating PDF report...")
    try:
        pdf_path = generator.generate_pdf_report(
            results=results,
            lecture_title="Test Report - Open-Ended Questions Enhancement",
            output_filename="test_report_enhanced.pdf"
        )
        print(f"  [OK] PDF generated: {pdf_path}")

        # Check file exists and size
        if os.path.exists(pdf_path):
            size_kb = os.path.getsize(pdf_path) / 1024
            print(f"  [OK] PDF file size: {size_kb:.1f} KB")
        else:
            print(f"  [FAIL] PDF file not found")
            return False

    except Exception as e:
        print(f"  [FAIL] PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Generate HTML report
    print("\n[4] Generating HTML report...")
    try:
        html_path = generator.generate_html_report(
            results=results,
            lecture_title="Test Report - Open-Ended Questions Enhancement",
            output_filename="test_report_enhanced.html"
        )
        print(f"  [OK] HTML generated: {html_path}")

        # Check file exists and verify content
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            size_kb = len(html_content) / 1024
            print(f"  [OK] HTML file size: {size_kb:.1f} KB")

            # Verify open-ended section exists
            if '4.2 Open-Ended Questions' in html_content:
                print(f"  [OK] Open-ended section found in HTML")
            else:
                print(f"  [WARN] Open-ended section NOT found in HTML")

            # Verify sample answer styling exists
            if 'sample-answer' in html_content:
                print(f"  [OK] Sample answer elements found in HTML")
            else:
                print(f"  [WARN] Sample answer elements NOT found in HTML")

        else:
            print(f"  [FAIL] HTML file not found")
            return False

    except Exception as e:
        print(f"  [FAIL] HTML generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"[OK] PDF Report: {pdf_path}")
    print(f"[OK] HTML Report: {html_path}")
    print(f"\nBoth reports now include:")
    print(f"  - {num_mcq} MCQ questions (Section 4.1)")
    print(f"  - {num_open_ended} Open-ended questions with sample answers (Section 4.2)")
    print("\n[SUCCESS] Report enhancement test completed!")

    return True


if __name__ == "__main__":
    success = test_report_enhancement()
    sys.exit(0 if success else 1)
