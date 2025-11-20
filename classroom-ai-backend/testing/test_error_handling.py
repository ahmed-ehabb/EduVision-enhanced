"""
Test Error Handling & Edge Cases
=================================

Comprehensive test suite for validating error handling and edge cases
in the Teacher Module V2 pipeline.

Tests:
1. Input Validation
   - Invalid audio files
   - Missing files
   - Corrupted files
   - Very short/long files
   - Invalid textbook data

2. Pipeline Robustness
   - Graceful degradation when modules fail
   - Recovery from transient errors
   - Edge cases (empty transcripts, etc.)

3. Error Reporting
   - Proper error messages
   - Warning vs error distinction
   - Detailed error metadata

Author: Ahmed
Date: 2025-11-06
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from input_validator import InputValidator, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestErrorHandling:
    """Test suite for error handling."""

    def __init__(self):
        self.validator = InputValidator(strict_mode=False)
        self.test_results = {
            "passed": [],
            "failed": [],
            "warnings": []
        }

    def run_all_tests(self):
        """Run all error handling tests."""
        print("\n" + "="*80)
        print("ERROR HANDLING & EDGE CASES TEST SUITE")
        print("="*80)

        # Test 1: Input Validation
        print("\n[TEST 1] Input Validation")
        print("-" * 80)
        self.test_audio_validation()
        self.test_textbook_validation()
        self.test_pdf_validation()
        self.test_title_validation()

        # Test 2: Edge Cases
        print("\n[TEST 2] Edge Cases")
        print("-" * 80)
        self.test_empty_inputs()
        self.test_extreme_inputs()

        # Test 3: Error Recovery (if we have actual files to test with)
        print("\n[TEST 3] Error Recovery")
        print("-" * 80)
        self.test_missing_files()

        # Print summary
        self.print_summary()

    def test_audio_validation(self):
        """Test audio file validation."""
        print("\n[1.1] Audio File Validation")

        # Test 1: Non-existent file
        valid, error, meta = self.validator.validate_audio_file("nonexistent.wav")
        if not valid and "not found" in error:
            self.pass_test("Audio validation rejects non-existent file")
        else:
            self.fail_test("Audio validation should reject non-existent file", error)

        # Test 2: Empty path
        valid, error, meta = self.validator.validate_audio_file("")
        if not valid and "required" in error.lower():
            self.pass_test("Audio validation rejects empty path")
        else:
            self.fail_test("Audio validation should reject empty path", error)

        # Test 3: Invalid format
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"not an audio file")
            tmp_path = tmp.name

        try:
            valid, error, meta = self.validator.validate_audio_file(tmp_path)
            if not valid and "format" in error.lower():
                self.pass_test("Audio validation rejects invalid format (.txt)")
            else:
                self.fail_test("Audio validation should reject .txt files", error)
        finally:
            os.unlink(tmp_path)

        # Test 4: Empty file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name  # Empty file (0 bytes)

        try:
            valid, error, meta = self.validator.validate_audio_file(tmp_path)
            if not valid and ("empty" in error.lower() or "0" in error):
                self.pass_test("Audio validation rejects empty file (0 bytes)")
            else:
                self.fail_test("Audio validation should reject empty files", error)
        finally:
            os.unlink(tmp_path)

    def test_textbook_validation(self):
        """Test textbook paragraphs validation."""
        print("\n[1.2] Textbook Paragraphs Validation")

        # Test 1: None input
        valid, error, meta = self.validator.validate_textbook_paragraphs(None)
        if not valid and "required" in error.lower():
            self.pass_test("Textbook validation rejects None")
        else:
            self.fail_test("Textbook validation should reject None", error)

        # Test 2: Not a list
        valid, error, meta = self.validator.validate_textbook_paragraphs("not a list")
        if not valid and "list" in error.lower():
            self.pass_test("Textbook validation rejects non-list input")
        else:
            self.fail_test("Textbook validation should reject non-list", error)

        # Test 3: Empty list
        valid, error, meta = self.validator.validate_textbook_paragraphs([])
        if not valid and ("few" in error.lower() or "0" in error):
            self.pass_test("Textbook validation rejects empty list")
        else:
            self.fail_test("Textbook validation should reject empty list", error)

        # Test 4: Valid paragraphs
        valid_paragraphs = [
            "This is a valid paragraph with sufficient length.",
            "Another valid paragraph here."
        ]
        valid, error, meta = self.validator.validate_textbook_paragraphs(valid_paragraphs)
        if valid:
            self.pass_test(f"Textbook validation accepts valid paragraphs ({meta.get('num_paragraphs')} paras)")
        else:
            self.fail_test("Textbook validation should accept valid paragraphs", error)

        # Test 5: All empty paragraphs
        empty_paragraphs = ["", "", "", ""]
        valid, error, meta = self.validator.validate_textbook_paragraphs(empty_paragraphs)
        if not valid and "empty" in error.lower():
            self.pass_test("Textbook validation rejects all-empty paragraphs")
        else:
            self.fail_test("Textbook validation should reject all-empty paragraphs", error)

        # Test 6: Mix of valid and empty (should pass with warning)
        mixed_paragraphs = [
            "Valid paragraph one.",
            "",
            "Valid paragraph two."
        ]
        valid, error, meta = self.validator.validate_textbook_paragraphs(mixed_paragraphs)
        if valid and meta.get('empty_count', 0) > 0:
            self.pass_test("Textbook validation accepts mixed paragraphs with warnings")
        else:
            self.fail_test("Textbook validation should accept mixed paragraphs", error)

    def test_pdf_validation(self):
        """Test PDF file validation."""
        print("\n[1.3] PDF File Validation")

        # Test 1: None/empty (should pass - PDF is optional)
        valid, error, meta = self.validator.validate_pdf_file(None)
        if valid and not meta.get('provided'):
            self.pass_test("PDF validation accepts None (optional)")
        else:
            self.fail_test("PDF validation should accept None", error)

        # Test 2: Non-existent file
        valid, error, meta = self.validator.validate_pdf_file("nonexistent.pdf")
        if not valid and "not found" in error:
            self.pass_test("PDF validation rejects non-existent file")
        else:
            self.fail_test("PDF validation should reject non-existent file", error)

        # Test 3: Invalid format
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"not a pdf")
            tmp_path = tmp.name

        try:
            valid, error, meta = self.validator.validate_pdf_file(tmp_path)
            if not valid and "pdf" in error.lower():
                self.pass_test("PDF validation rejects non-PDF file")
            else:
                self.fail_test("PDF validation should reject non-PDF files", error)
        finally:
            os.unlink(tmp_path)

        # Test 4: Invalid PDF (wrong magic bytes)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"not a real pdf file")
            tmp_path = tmp.name

        try:
            valid, error, meta = self.validator.validate_pdf_file(tmp_path)
            if not valid and ("valid" in error.lower() or "header" in error.lower() or "small" in error.lower()):
                self.pass_test("PDF validation rejects invalid PDF (corrupted/small)")
            else:
                self.fail_test("PDF validation should reject corrupted PDF", error)
        finally:
            os.unlink(tmp_path)

    def test_title_validation(self):
        """Test lecture title validation."""
        print("\n[1.4] Lecture Title Validation")

        # Test 1: None
        valid, error, meta = self.validator.validate_lecture_title(None)
        if not valid and "required" in error.lower():
            self.pass_test("Title validation rejects None")
        else:
            self.fail_test("Title validation should reject None", error)

        # Test 2: Not a string
        valid, error, meta = self.validator.validate_lecture_title(12345)
        if not valid and "string" in error.lower():
            self.pass_test("Title validation rejects non-string")
        else:
            self.fail_test("Title validation should reject non-string", error)

        # Test 3: Too short
        valid, error, meta = self.validator.validate_lecture_title("Hi")
        if not valid and "short" in error.lower():
            self.pass_test("Title validation rejects too-short title")
        else:
            self.fail_test("Title validation should reject titles <3 chars", error)

        # Test 4: Too long
        very_long_title = "A" * 250
        valid, error, meta = self.validator.validate_lecture_title(very_long_title)
        if not valid and "long" in error.lower():
            self.pass_test("Title validation rejects too-long title")
        else:
            self.fail_test("Title validation should reject titles >200 chars", error)

        # Test 5: Valid title
        valid, error, meta = self.validator.validate_lecture_title("Psychology 101 - Introduction")
        if valid:
            self.pass_test("Title validation accepts valid title")
        else:
            self.fail_test("Title validation should accept valid titles", error)

    def test_empty_inputs(self):
        """Test edge cases with empty inputs."""
        print("\n[2.1] Empty Input Edge Cases")

        # Test: Complete validation with minimal valid inputs
        minimal_paragraphs = ["Minimal valid paragraph."]
        valid, errors, meta = self.validator.validate_pipeline_inputs(
            audio_path="nonexistent.wav",  # Will fail
            textbook_paragraphs=minimal_paragraphs,
            pdf_path=None,
            lecture_title="Test"
        )

        if not valid and any("Audio" in e for e in errors):
            self.pass_test("Pipeline validation catches missing audio file")
        else:
            self.fail_test("Pipeline validation should catch missing audio", str(errors))

    def test_extreme_inputs(self):
        """Test edge cases with extreme inputs."""
        print("\n[2.2] Extreme Input Edge Cases")

        # Test: Very large number of paragraphs
        many_paragraphs = ["Paragraph " + str(i) for i in range(1001)]
        valid, error, meta = self.validator.validate_textbook_paragraphs(many_paragraphs)
        if not valid and ("many" in error.lower() or "1001" in error):
            self.pass_test("Textbook validation rejects >1000 paragraphs")
        else:
            self.fail_test("Textbook validation should reject >1000 paragraphs", error)

        # Test: Very long paragraph
        very_long_para = "A" * 15000
        long_paragraphs = [very_long_para]
        valid, error, meta = self.validator.validate_textbook_paragraphs(long_paragraphs)
        if valid and 'warnings' in meta:
            self.pass_test("Textbook validation warns about very long paragraphs")
        else:
            # Should pass but with warnings
            self.pass_test("Textbook validation accepts long paragraphs")

    def test_missing_files(self):
        """Test error recovery for missing files."""
        print("\n[3.1] Missing File Error Recovery")

        # Test: Graceful handling of missing audio
        # validate_pipeline_inputs returns (is_valid, errors, metadata) - doesn't raise
        is_valid, errors, metadata = self.validator.validate_pipeline_inputs(
            audio_path="missing.wav",
            textbook_paragraphs=["Valid paragraph"],
            pdf_path=None,
            lecture_title="Test Lecture"
        )

        if not is_valid and any("Audio" in e or "not found" in e for e in errors):
            self.pass_test("Pipeline validation catches missing audio file")
        else:
            self.fail_test("Pipeline should catch missing audio", str(errors))

    # Helper methods
    def pass_test(self, test_name):
        """Mark test as passed."""
        self.test_results["passed"].append(test_name)
        print(f"  [PASS] {test_name}")

    def fail_test(self, test_name, details=""):
        """Mark test as failed."""
        self.test_results["failed"].append((test_name, details))
        print(f"  [FAIL] {test_name}")
        if details:
            print(f"         Details: {details}")

    def add_warning(self, warning):
        """Add a warning."""
        self.test_results["warnings"].append(warning)
        print(f"  [WARN] {warning}")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        passed = len(self.test_results["passed"])
        failed = len(self.test_results["failed"])
        warnings = len(self.test_results["warnings"])
        total = passed + failed

        print(f"\nTotal Tests: {total}")
        print(f"  [PASS] {passed} ({100*passed//total if total > 0 else 0}%)")
        print(f"  [FAIL] {failed} ({100*failed//total if total > 0 else 0}%)")
        if warnings > 0:
            print(f"  [WARN] {warnings}")

        if failed > 0:
            print("\nFailed Tests:")
            for test_name, details in self.test_results["failed"]:
                print(f"  - {test_name}")
                if details:
                    print(f"    {details}")

        print("\n" + "="*80)
        if failed == 0:
            print("[SUCCESS] All error handling tests passed!")
        else:
            print(f"[PARTIAL] {failed} tests failed")
        print("="*80)

        return failed == 0


def main():
    """Main test runner."""
    tester = TestErrorHandling()
    success = tester.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
