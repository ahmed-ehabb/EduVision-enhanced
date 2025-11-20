"""
Input Validation Module
=======================

Validates all inputs to the Teacher Module V2 pipeline:
- Audio files (format, size, duration, quality)
- Text inputs (textbook paragraphs, lecture titles)
- PDF files (for quiz RAG)
- Configuration parameters

Author: Ahmed
Date: 2025-11-06
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import mimetypes

logger = logging.getLogger(__name__)

# Configuration constants
AUDIO_CONFIG = {
    'supported_formats': ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac'],
    'max_file_size_mb': 500,  # 500 MB max
    'min_duration_seconds': 30,  # 30 seconds minimum
    'max_duration_seconds': 7200,  # 2 hours maximum
    'min_sample_rate': 8000,  # 8kHz minimum
    'recommended_sample_rate': 16000  # 16kHz recommended for Whisper
}

PDF_CONFIG = {
    'supported_formats': ['.pdf'],
    'max_file_size_mb': 50,  # 50 MB max for PDFs
    'min_file_size_bytes': 100  # At least 100 bytes
}

TEXT_CONFIG = {
    'max_paragraph_length': 10000,  # 10k characters per paragraph
    'min_paragraph_length': 10,  # At least 10 characters
    'max_paragraphs': 1000,  # Max 1000 paragraphs
    'min_paragraphs': 1,  # At least 1 paragraph
    'max_title_length': 200,
    'min_title_length': 3
}


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, field: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.field = field
        self.details = details or {}
        super().__init__(self.message)


class InputValidator:
    """Validates all inputs to the Teacher Module pipeline."""

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.

        Args:
            strict_mode: If True, apply stricter validation rules
        """
        self.strict_mode = strict_mode
        logger.info(f"[InputValidator] Initialized (strict_mode={strict_mode})")

    def validate_audio_file(self, audio_path: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate audio file for ASR processing.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        metadata = {}

        # Check if path is provided
        if not audio_path:
            return False, "Audio path is required", metadata

        # Convert to Path object
        audio_path = Path(audio_path)

        # Check if file exists
        if not audio_path.exists():
            return False, f"Audio file not found: {audio_path}", metadata

        # Check if it's a file (not directory)
        if not audio_path.is_file():
            return False, f"Path is not a file: {audio_path}", metadata

        # Check file extension
        file_ext = audio_path.suffix.lower()
        if file_ext not in AUDIO_CONFIG['supported_formats']:
            return False, (
                f"Unsupported audio format: {file_ext}. "
                f"Supported: {', '.join(AUDIO_CONFIG['supported_formats'])}"
            ), metadata

        # Check file size
        file_size_bytes = audio_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 ** 2)
        metadata['file_size_mb'] = round(file_size_mb, 2)

        if file_size_bytes == 0:
            return False, "Audio file is empty (0 bytes)", metadata

        if file_size_mb > AUDIO_CONFIG['max_file_size_mb']:
            return False, (
                f"Audio file too large: {file_size_mb:.2f} MB "
                f"(max: {AUDIO_CONFIG['max_file_size_mb']} MB)"
            ), metadata

        # Try to get audio duration and sample rate (requires librosa or similar)
        try:
            duration, sample_rate = self._get_audio_info(audio_path)
            metadata['duration_seconds'] = round(duration, 2)
            metadata['sample_rate'] = sample_rate

            # Check duration
            if duration < AUDIO_CONFIG['min_duration_seconds']:
                return False, (
                    f"Audio too short: {duration:.1f}s "
                    f"(min: {AUDIO_CONFIG['min_duration_seconds']}s)"
                ), metadata

            if duration > AUDIO_CONFIG['max_duration_seconds']:
                return False, (
                    f"Audio too long: {duration:.1f}s "
                    f"(max: {AUDIO_CONFIG['max_duration_seconds']}s = 2 hours)"
                ), metadata

            # Check sample rate
            if sample_rate < AUDIO_CONFIG['min_sample_rate']:
                return False, (
                    f"Sample rate too low: {sample_rate} Hz "
                    f"(min: {AUDIO_CONFIG['min_sample_rate']} Hz)"
                ), metadata

            # Warning for non-optimal sample rate (not an error)
            if sample_rate != AUDIO_CONFIG['recommended_sample_rate']:
                metadata['warning'] = (
                    f"Sample rate {sample_rate} Hz. "
                    f"Recommended: {AUDIO_CONFIG['recommended_sample_rate']} Hz for best results"
                )

        except Exception as e:
            # If we can't read audio metadata, log warning but don't fail
            # (ASR module will handle actual audio processing errors)
            logger.warning(f"Could not read audio metadata: {e}")
            metadata['warning'] = f"Could not verify audio properties: {str(e)}"

        # All checks passed
        metadata['format'] = file_ext
        metadata['valid'] = True
        return True, None, metadata

    def validate_textbook_paragraphs(
        self,
        paragraphs: List[str]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate textbook paragraphs for content alignment.

        Args:
            paragraphs: List of textbook paragraph strings

        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        metadata = {}

        # Check if paragraphs provided
        if paragraphs is None:
            return False, "Textbook paragraphs are required", metadata

        if not isinstance(paragraphs, list):
            return False, f"Textbook paragraphs must be a list, got {type(paragraphs)}", metadata

        # Check number of paragraphs
        num_paragraphs = len(paragraphs)
        metadata['num_paragraphs'] = num_paragraphs

        if num_paragraphs < TEXT_CONFIG['min_paragraphs']:
            return False, (
                f"Too few paragraphs: {num_paragraphs} "
                f"(min: {TEXT_CONFIG['min_paragraphs']})"
            ), metadata

        if num_paragraphs > TEXT_CONFIG['max_paragraphs']:
            return False, (
                f"Too many paragraphs: {num_paragraphs} "
                f"(max: {TEXT_CONFIG['max_paragraphs']})"
            ), metadata

        # Check each paragraph
        empty_count = 0
        too_short_count = 0
        too_long_count = 0
        total_chars = 0

        for i, para in enumerate(paragraphs):
            if not isinstance(para, str):
                return False, f"Paragraph {i} is not a string: {type(para)}", metadata

            para_len = len(para.strip())
            total_chars += para_len

            if para_len == 0:
                empty_count += 1
            elif para_len < TEXT_CONFIG['min_paragraph_length']:
                too_short_count += 1
            elif para_len > TEXT_CONFIG['max_paragraph_length']:
                too_long_count += 1

        # Check for too many empty paragraphs
        if empty_count > num_paragraphs * 0.5:  # >50% empty
            return False, f"Too many empty paragraphs: {empty_count}/{num_paragraphs}", metadata

        # Warnings for quality issues (not errors)
        warnings = []
        if too_short_count > 0:
            warnings.append(f"{too_short_count} paragraphs shorter than {TEXT_CONFIG['min_paragraph_length']} chars")
        if too_long_count > 0:
            warnings.append(f"{too_long_count} paragraphs longer than {TEXT_CONFIG['max_paragraph_length']} chars")
        if empty_count > 0:
            warnings.append(f"{empty_count} empty paragraphs will be ignored")

        metadata['total_characters'] = total_chars
        metadata['avg_paragraph_length'] = round(total_chars / num_paragraphs, 1) if num_paragraphs > 0 else 0
        metadata['empty_count'] = empty_count
        if warnings:
            metadata['warnings'] = warnings

        return True, None, metadata

    def validate_pdf_file(self, pdf_path: Optional[str]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate PDF file for quiz RAG.

        Args:
            pdf_path: Path to PDF file (can be None)

        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        metadata = {}

        # PDF is optional
        if pdf_path is None or pdf_path == "":
            metadata['provided'] = False
            return True, None, metadata

        metadata['provided'] = True

        # Convert to Path object
        pdf_path = Path(pdf_path)

        # Check if file exists
        if not pdf_path.exists():
            return False, f"PDF file not found: {pdf_path}", metadata

        # Check if it's a file
        if not pdf_path.is_file():
            return False, f"Path is not a file: {pdf_path}", metadata

        # Check file extension
        file_ext = pdf_path.suffix.lower()
        if file_ext not in PDF_CONFIG['supported_formats']:
            return False, f"Not a PDF file: {file_ext}", metadata

        # Check file size
        file_size_bytes = pdf_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 ** 2)
        metadata['file_size_mb'] = round(file_size_mb, 2)

        if file_size_bytes < PDF_CONFIG['min_file_size_bytes']:
            return False, f"PDF file too small: {file_size_bytes} bytes (possibly corrupted)", metadata

        if file_size_mb > PDF_CONFIG['max_file_size_mb']:
            return False, (
                f"PDF file too large: {file_size_mb:.2f} MB "
                f"(max: {PDF_CONFIG['max_file_size_mb']} MB)"
            ), metadata

        # Try to verify it's a valid PDF (check magic bytes)
        try:
            with open(pdf_path, 'rb') as f:
                header = f.read(5)
                if header != b'%PDF-':
                    return False, "File does not appear to be a valid PDF (invalid header)", metadata
        except Exception as e:
            return False, f"Could not read PDF file: {e}", metadata

        metadata['valid'] = True
        return True, None, metadata

    def validate_lecture_title(self, title: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate lecture title.

        Args:
            title: Lecture title string

        Returns:
            Tuple of (is_valid, error_message, metadata)
        """
        metadata = {}

        if title is None:
            return False, "Lecture title is required", metadata

        if not isinstance(title, str):
            return False, f"Lecture title must be a string, got {type(title)}", metadata

        title_len = len(title.strip())
        metadata['length'] = title_len

        if title_len < TEXT_CONFIG['min_title_length']:
            return False, f"Lecture title too short: {title_len} chars (min: {TEXT_CONFIG['min_title_length']})", metadata

        if title_len > TEXT_CONFIG['max_title_length']:
            return False, f"Lecture title too long: {title_len} chars (max: {TEXT_CONFIG['max_title_length']})", metadata

        metadata['valid'] = True
        return True, None, metadata

    def validate_pipeline_inputs(
        self,
        audio_path: str,
        textbook_paragraphs: List[str],
        pdf_path: Optional[str] = None,
        lecture_title: str = "Lecture"
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate all inputs for the complete pipeline.

        Args:
            audio_path: Path to audio file
            textbook_paragraphs: List of textbook paragraphs
            pdf_path: Optional path to PDF file
            lecture_title: Lecture title

        Returns:
            Tuple of (is_valid, error_messages, all_metadata)
        """
        errors = []
        all_metadata = {}

        # Validate audio
        audio_valid, audio_error, audio_meta = self.validate_audio_file(audio_path)
        all_metadata['audio'] = audio_meta
        if not audio_valid:
            errors.append(f"Audio validation failed: {audio_error}")

        # Validate textbook paragraphs
        textbook_valid, textbook_error, textbook_meta = self.validate_textbook_paragraphs(textbook_paragraphs)
        all_metadata['textbook'] = textbook_meta
        if not textbook_valid:
            errors.append(f"Textbook validation failed: {textbook_error}")

        # Validate PDF (optional)
        pdf_valid, pdf_error, pdf_meta = self.validate_pdf_file(pdf_path)
        all_metadata['pdf'] = pdf_meta
        if not pdf_valid:
            errors.append(f"PDF validation failed: {pdf_error}")

        # Validate title
        title_valid, title_error, title_meta = self.validate_lecture_title(lecture_title)
        all_metadata['title'] = title_meta
        if not title_valid:
            errors.append(f"Title validation failed: {title_error}")

        is_valid = len(errors) == 0
        all_metadata['validation_passed'] = is_valid

        if is_valid:
            logger.info("[InputValidator] All inputs validated successfully")
        else:
            logger.error(f"[InputValidator] Validation failed with {len(errors)} errors")

        return is_valid, errors, all_metadata

    def _get_audio_info(self, audio_path: Path) -> Tuple[float, int]:
        """
        Get audio duration and sample rate.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (duration_seconds, sample_rate)
        """
        # Try multiple methods to get audio info
        try:
            # Method 1: Use librosa (preferred, but optional dependency)
            import librosa
            duration = librosa.get_duration(path=str(audio_path))
            y, sr = librosa.load(str(audio_path), sr=None, duration=1.0)  # Just load 1 second for speed
            return duration, sr
        except ImportError:
            pass  # librosa not installed

        try:
            # Method 2: Use soundfile (faster than librosa)
            import soundfile as sf
            info = sf.info(str(audio_path))
            return info.duration, info.samplerate
        except ImportError:
            pass  # soundfile not installed

        try:
            # Method 3: Use pydub (requires ffmpeg)
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(audio_path))
            duration = len(audio) / 1000.0  # milliseconds to seconds
            sample_rate = audio.frame_rate
            return duration, sample_rate
        except ImportError:
            pass  # pydub not installed

        # If no library available, raise exception
        raise ImportError(
            "Cannot read audio metadata. Please install one of: librosa, soundfile, or pydub"
        )


def validate_inputs(
    audio_path: str,
    textbook_paragraphs: List[str],
    pdf_path: Optional[str] = None,
    lecture_title: str = "Lecture",
    strict_mode: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to validate all inputs.

    Args:
        audio_path: Path to audio file
        textbook_paragraphs: List of textbook paragraphs
        pdf_path: Optional path to PDF
        lecture_title: Lecture title
        strict_mode: Enable strict validation

    Returns:
        Validation results dictionary

    Raises:
        ValidationError: If validation fails
    """
    validator = InputValidator(strict_mode=strict_mode)
    is_valid, errors, metadata = validator.validate_pipeline_inputs(
        audio_path=audio_path,
        textbook_paragraphs=textbook_paragraphs,
        pdf_path=pdf_path,
        lecture_title=lecture_title
    )

    if not is_valid:
        raise ValidationError(
            message=f"Input validation failed: {'; '.join(errors)}",
            details={'errors': errors, 'metadata': metadata}
        )

    return metadata


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test validation
    validator = InputValidator()

    # Test audio validation
    print("\n=== Testing Audio Validation ===")
    valid, error, meta = validator.validate_audio_file("test.wav")
    print(f"Valid: {valid}, Error: {error}, Metadata: {meta}")

    # Test textbook validation
    print("\n=== Testing Textbook Validation ===")
    paragraphs = ["This is paragraph 1", "This is paragraph 2"]
    valid, error, meta = validator.validate_textbook_paragraphs(paragraphs)
    print(f"Valid: {valid}, Error: {error}, Metadata: {meta}")

    print("\n[InputValidator] Test complete")
