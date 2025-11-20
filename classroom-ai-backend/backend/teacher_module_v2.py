"""
Teacher Module V2 - Unified Lecture Evaluation System

Orchestrates all 7 AI models to provide complete lecture evaluation:
1. ASR - Transcription
2. Engagement - Audio analysis
3. Content Alignment - Textbook comparison
4. Translation - Arabic to English
5. Notes - Summary generation
6. Quiz - Question generation
7. Punctuation - Text restoration (optional)

Optimized for RTX 3050 (4GB VRAM) with sequential model loading.

Author: Ahmed
"""

import os
import gc
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Import validation and error handling
from input_validator import InputValidator, ValidationError
from error_handler import handle_error, ErrorSeverity, retry_with_backoff, RetryConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class TeacherModuleV2:
    """
    Unified teacher evaluation system.

    Manages all 7 AI models with sequential loading to stay within 4GB VRAM.
    """

    def __init__(
        self,
        use_punctuation: bool = False,
        use_translation: bool = True,
        enable_quiz: bool = True,
        enable_notes: bool = True
    ):
        """
        Initialize Teacher Module.

        Args:
            use_punctuation: Enable punctuation restoration (very slow)
            use_translation: Enable translation for Arabic text
            enable_quiz: Generate quiz questions
            enable_notes: Generate lecture notes
        """
        self.use_punctuation = use_punctuation
        self.use_translation = use_translation
        self.enable_quiz = enable_quiz
        self.enable_notes = enable_notes

        # Model instances (lazy loaded)
        self.asr = None
        self.engagement = None
        self.alignment = None
        self.translator = None
        self.notes_gen = None
        self.quiz_gen = None

        # Input validator
        self.validator = InputValidator(strict_mode=False)

        logger.info("[TeacherModule] Initialized")

    def process_lecture(
        self,
        audio_path: str,
        textbook_paragraphs: List[str],
        pdf_path: Optional[str] = None,
        lecture_title: str = "Lecture",
        generate_reports: bool = True,
        report_formats: List[str] = ['pdf', 'html']
    ) -> Dict[str, Any]:
        """
        Complete lecture evaluation pipeline.

        Args:
            audio_path: Path to lecture audio file
            textbook_paragraphs: List of textbook paragraph strings
            pdf_path: Optional path to PDF for quiz RAG
            lecture_title: Title for the lecture
            generate_reports: Generate PDF/HTML reports
            report_formats: List of report formats ('pdf', 'html')

        Returns:
            Complete evaluation results dictionary
        """
        logger.info("="*80)
        logger.info(f"[TeacherModule] Starting lecture evaluation: {lecture_title}")
        logger.info("="*80)

        start_time = time.time()
        results = {
            "lecture_title": lecture_title,
            "audio_path": audio_path,
            "timestamp": datetime.now().isoformat(),
            "processing_steps": [],
            "errors": [],
            "warnings": []
        }

        # STEP 0: Input Validation
        logger.info("\n[Step 0/7] Validating Inputs...")
        try:
            is_valid, validation_errors, validation_metadata = self.validator.validate_pipeline_inputs(
                audio_path=audio_path,
                textbook_paragraphs=textbook_paragraphs,
                pdf_path=pdf_path,
                lecture_title=lecture_title
            )

            results["validation"] = validation_metadata

            if not is_valid:
                # Validation failed - return early
                logger.error(f"[TeacherModule] Input validation failed:")
                for error in validation_errors:
                    logger.error(f"  - {error}")

                results["success"] = False
                results["errors"] = validation_errors
                results["total_processing_time"] = round(time.time() - start_time, 2)
                return results

            # Log validation metadata
            if 'audio' in validation_metadata:
                audio_meta = validation_metadata['audio']
                if 'duration_seconds' in audio_meta:
                    logger.info(f"  Audio: {audio_meta['duration_seconds']}s, {audio_meta.get('file_size_mb')}MB")
                if 'warning' in audio_meta:
                    logger.warning(f"  {audio_meta['warning']}")
                    results["warnings"].append(audio_meta['warning'])

            if 'textbook' in validation_metadata:
                tb_meta = validation_metadata['textbook']
                logger.info(f"  Textbook: {tb_meta.get('num_paragraphs')} paragraphs, {tb_meta.get('total_characters')} chars")
                if 'warnings' in tb_meta:
                    for warning in tb_meta['warnings']:
                        logger.warning(f"  {warning}")
                        results["warnings"].append(warning)

            logger.info("  ✓ All inputs validated successfully")

        except Exception as e:
            logger.error(f"[TeacherModule] Validation error: {e}")
            results["success"] = False
            results["errors"].append(f"Validation error: {str(e)}")
            results["total_processing_time"] = round(time.time() - start_time, 2)
            return results

        try:
            # Step 1: Transcription (ASR)
            logger.info("\n[Step 1/7] Transcription (ASR)...")
            transcript_result = self._step_transcribe(audio_path)
            results["transcript"] = transcript_result
            results["processing_steps"].append("transcription")

            if not transcript_result.get("success"):
                error_msg = transcript_result.get("error", "Unknown transcription error")
                results["errors"].append(f"Transcription failed: {error_msg}")
                results["success"] = False
                handle_error(Exception(error_msg), ErrorSeverity.CRITICAL, "ASR Transcription")
                return results

            transcript_text = transcript_result["text"]
            transcript_segments = transcript_result.get("segments", [transcript_text])

            # Edge case: Check for empty or very short transcript
            if not transcript_text or len(transcript_text.strip()) < 50:
                logger.warning("  ⚠ Transcript is very short or empty")
                results["warnings"].append("Transcript is unusually short - lecture may be too quiet or short")
                # Continue anyway, but log warning

            # Step 2: Engagement Analysis
            logger.info("\n[Step 2/7] Engagement Analysis...")
            try:
                engagement_result = self._step_engagement(audio_path, transcript_segments)
                results["engagement"] = engagement_result
                results["processing_steps"].append("engagement")

                # Check for failed engagement analysis (graceful degradation)
                if not engagement_result.get("success"):
                    error_msg = engagement_result.get("error", "Unknown engagement error")
                    logger.warning(f"  ⚠ Engagement analysis failed: {error_msg}")
                    results["warnings"].append(f"Engagement analysis skipped: {error_msg}")
                    handle_error(Exception(error_msg), ErrorSeverity.MEDIUM, "Engagement Analysis")
                    # Continue with pipeline anyway

            except Exception as e:
                logger.error(f"  ✗ Engagement analysis encountered unexpected error: {e}")
                results["engagement"] = {"success": False, "error": str(e)}
                results["warnings"].append(f"Engagement analysis failed: {str(e)}")
                handle_error(e, ErrorSeverity.MEDIUM, "Engagement Analysis")
                # Continue with pipeline

            # Step 3: Content Alignment
            logger.info("\n[Step 3/7] Content Alignment...")
            try:
                alignment_result = self._step_alignment(transcript_segments, textbook_paragraphs)
                results["content_alignment"] = alignment_result
                results["processing_steps"].append("content_alignment")

                # Check for failed alignment (graceful degradation)
                if not alignment_result.get("success"):
                    error_msg = alignment_result.get("error", "Unknown alignment error")
                    logger.warning(f"  ⚠ Content alignment failed: {error_msg}")
                    results["warnings"].append(f"Content alignment skipped: {error_msg}")
                    handle_error(Exception(error_msg), ErrorSeverity.MEDIUM, "Content Alignment")
                    # Continue with pipeline anyway

            except Exception as e:
                logger.error(f"  ✗ Content alignment encountered unexpected error: {e}")
                results["content_alignment"] = {"success": False, "error": str(e)}
                results["warnings"].append(f"Content alignment failed: {str(e)}")
                handle_error(e, ErrorSeverity.MEDIUM, "Content Alignment")
                # Continue with pipeline

            # Step 4: Translation (if enabled and needed)
            if self.use_translation and self._needs_translation(transcript_text):
                logger.info("\n[Step 4/7] Translation...")
                translation_result = self._step_translate(transcript_text)
                results["translation"] = translation_result
                results["processing_steps"].append("translation")
                transcript_for_generation = translation_result.get("translated_text", transcript_text)
            else:
                logger.info("\n[Step 4/7] Translation (skipped - not needed)")
                results["translation"] = {"skipped": True}
                transcript_for_generation = transcript_text

            # Step 5: Notes Generation
            if self.enable_notes:
                logger.info("\n[Step 5/7] Notes Generation...")
                notes_result = self._step_notes(transcript_for_generation, lecture_title)
                results["notes"] = notes_result
                results["processing_steps"].append("notes")
            else:
                logger.info("\n[Step 5/7] Notes Generation (disabled)")
                results["notes"] = {"disabled": True}

            # Step 6: Quiz Generation
            if self.enable_quiz:
                logger.info("\n[Step 6/7] Quiz Generation...")
                quiz_result = self._step_quiz(transcript_for_generation, pdf_path)
                results["quiz"] = quiz_result
                results["processing_steps"].append("quiz")
            else:
                logger.info("\n[Step 6/7] Quiz Generation (disabled)")
                results["quiz"] = {"disabled": True}

            # Step 7: Final Summary
            logger.info("\n[Step 7/7] Generating Summary...")
            results["summary"] = self._generate_summary(results)

            # Calculate total time
            total_time = time.time() - start_time
            results["total_processing_time"] = round(total_time, 2)
            results["total_time"] = round(total_time / 60, 2)  # minutes for report
            results["transcript_length"] = len(transcript_text)

            # Mark as successful (warnings are OK, only errors prevent success)
            results["success"] = True
            results["has_warnings"] = len(results["warnings"]) > 0

            logger.info("="*80)
            logger.info(f"[TeacherModule] Evaluation Complete in {total_time:.2f}s")
            if results["has_warnings"]:
                logger.info(f"[TeacherModule] Completed with {len(results['warnings'])} warnings")
            logger.info("="*80)

            # Generate reports if requested
            if generate_reports and report_formats:
                logger.info("\n[Step 8/8] Generating Reports...")
                try:
                    from report_generator import ReportGenerator

                    report_gen = ReportGenerator()
                    report_paths = report_gen.generate_reports(
                        results,
                        lecture_title,
                        report_formats
                    )
                    results["report_paths"] = report_paths

                    logger.info("  Reports generated:")
                    for fmt, path in report_paths.items():
                        logger.info(f"    {fmt.upper()}: {path}")

                except Exception as e:
                    logger.error(f"  Failed to generate reports: {e}")
                    results["report_error"] = str(e)

            return results

        except Exception as e:
            logger.error(f"[TeacherModule] Fatal error during pipeline execution: {e}")
            import traceback
            traceback.print_exc()

            # Log detailed error information
            handle_error(e, ErrorSeverity.CRITICAL, "TeacherModule Pipeline")

            # Add error to results
            results["errors"].append(f"Fatal pipeline error: {str(e)}")
            results["success"] = False
            results["fatal_error"] = True
            results["total_processing_time"] = round(time.time() - start_time, 2)

            logger.info("="*80)
            logger.error(f"[TeacherModule] Pipeline FAILED after {results['total_processing_time']:.2f}s")
            logger.info("="*80)

            return results

    def _step_transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Step 1: Transcribe audio using ASR."""
        try:
            from asr_module import ASRModule

            # Load ASR
            logger.info("  Loading ASR model (INT8)...")
            try:
                self.asr = ASRModule(
                    model_id="ahmedheakl/arazn-whisper-small-v2",
                    enable_punctuation=self.use_punctuation,
                    compute_type="int8"
                )
            except Exception as model_load_error:
                logger.error(f"  ✗ Failed to load ASR model: {model_load_error}")
                handle_error(model_load_error, ErrorSeverity.CRITICAL, "ASR Model Loading")
                return {"success": False, "error": f"Model loading failed: {str(model_load_error)}"}

            # Transcribe
            logger.info(f"  Transcribing: {audio_path}")
            try:
                result = self.asr.transcribe(audio_path, add_punctuation=self.use_punctuation)
            except Exception as transcribe_error:
                logger.error(f"  ✗ Transcription failed: {transcribe_error}")
                handle_error(transcribe_error, ErrorSeverity.HIGH, "ASR Transcription")

                # Attempt cleanup
                self._unload_model("asr")

                return {"success": False, "error": f"Transcription failed: {str(transcribe_error)}"}

            if result:
                # Parse into segments
                from content_alignment_v2 import segment_transcript
                text = result.get("punctuated_text", result.get("text", ""))

                # Edge case: Empty transcript
                if not text or len(text.strip()) == 0:
                    logger.warning("  ⚠ Transcription produced empty text")
                    return {
                        "success": False,
                        "error": "Transcription produced empty result - audio may be silent or corrupted"
                    }

                segments = segment_transcript(text)

                output = {
                    "success": True,
                    "text": text,
                    "segments": segments,
                    "processing_time": result.get("processing_time", 0),
                    "model": result.get("model", ""),
                    "num_segments": len(segments)
                }

                logger.info(f"  ✓ Transcribed: {len(text)} chars, {len(segments)} segments")

                # Unload ASR
                self._unload_model("asr")

                return output
            else:
                logger.error("  ✗ Transcription returned None or empty result")
                return {"success": False, "error": "Transcription returned no result"}

        except ImportError as import_error:
            logger.error(f"  ✗ Missing dependency for ASR: {import_error}")
            handle_error(import_error, ErrorSeverity.CRITICAL, "ASR Dependencies")
            return {"success": False, "error": f"Missing ASR dependencies: {str(import_error)}"}

        except Exception as e:
            logger.error(f"  ✗ Unexpected transcription error: {e}")
            import traceback
            traceback.print_exc()
            handle_error(e, ErrorSeverity.CRITICAL, "ASR Transcription")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    def _step_engagement(self, audio_path: str, transcript_segments: List[str]) -> Dict[str, Any]:
        """Step 2: Analyze engagement from audio."""
        try:
            from engagement_analyzer_v2 import EngagementAnalyzerV2

            logger.info("  Analyzing engagement (CPU-based)...")
            self.engagement = EngagementAnalyzerV2()

            result = self.engagement.analyze_audio(audio_path, transcript_segments)

            logger.info(f"  ✓ Engagement Score: {result['engagement_score']:.2f}%")

            # Engagement runs on CPU, no need to unload
            return {
                "success": True,
                "engagement_score": result["engagement_score"],
                "statistics": result["statistics"],
                "results": result["results"]
            }

        except Exception as e:
            logger.error(f"  ✗ Engagement analysis failed: {e}")
            return {"success": False, "error": str(e)}

    def _step_alignment(
        self,
        transcript_segments: List[str],
        textbook_paragraphs: List[str]
    ) -> Dict[str, Any]:
        """Step 3: Align transcript with textbook content."""
        try:
            from content_alignment_v2 import ContentAlignmentAnalyzer

            logger.info("  Loading Content Alignment (SBERT)...")
            self.alignment = ContentAlignmentAnalyzer()

            # Load textbook
            logger.info(f"  Loading {len(textbook_paragraphs)} textbook paragraphs...")
            self.alignment.load_textbook_content(textbook_paragraphs)

            # Analyze
            logger.info(f"  Analyzing {len(transcript_segments)} segments...")
            result = self.alignment.analyze_transcript(transcript_segments)

            logger.info(f"  ✓ Coverage Score: {result['coverage_score']:.2f}%")

            # Unload alignment
            self._unload_model("alignment")

            return {
                "success": True,
                "coverage_score": result["coverage_score"],
                "coverage_percentages": result["coverage_percentages"],
                "coverage_percentage": result.get("coverage_percentage", 0),
                "num_covered_topics": result.get("num_covered_topics", 0),
                "num_total_topics": result.get("num_total_topics", 0),
                "covered_topics": result.get("covered_topics", []),
                "feedback": result["feedback"],
                "results": result["results"]
            }

        except Exception as e:
            logger.error(f"  ✗ Content alignment failed: {e}")
            return {"success": False, "error": str(e)}

    def _step_translate(self, text: str) -> Dict[str, Any]:
        """Step 4: Translate Arabic to English."""
        try:
            from translation_module import translate_text, TranslationModel

            logger.info("  Loading Translation model (GGUF)...")
            self.translator = TranslationModel()

            logger.info("  Translating text...")
            result = translate_text(text, self.translator)

            logger.info(f"  ✓ Translation complete")

            # Translator manages its own memory
            return {
                "success": True,
                "translated_text": result.get("translated_text", ""),
                "processing_time": result.get("processing_time", 0),
                "language_stats": result.get("language_stats", {})
            }

        except Exception as e:
            logger.error(f"  ✗ Translation failed: {e}")
            return {"success": False, "error": str(e), "translated_text": text}

    def _step_notes(self, transcript: str, title: str) -> Dict[str, Any]:
        """Step 5: Generate lecture notes using original algorithm."""
        try:
            from notes_generator import LectureNotesGenerator

            logger.info("  Loading Notes Generator (original algorithm with deduplication)...")
            self.notes_gen = LectureNotesGenerator()

            logger.info("  Generating notes...")
            # Use original algorithm: chunking, deduplication, grammar correction
            notes_markdown = self.notes_gen.generate_notes(transcript, subject=title)

            # Extract bullet points from markdown (format: "* bullet text")
            bullet_points = []
            for line in notes_markdown.split('\n'):
                line = line.strip()
                if line.startswith('* '):
                    bullet_points.append(line[2:].strip())  # Remove "* " prefix
                elif line.startswith('- '):
                    bullet_points.append(line[2:].strip())  # Also support "- " format

            num_bullets = len(bullet_points)
            logger.info(f"  ✓ Generated {num_bullets} bullet points (with deduplication)")

            # Unload notes generator with aggressive cleanup
            self.notes_gen = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Extra aggressive cleanup before loading large Quiz model (2GB+)
            import time
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(2)  # Give system time to fully release memory
            logger.info("  Memory cleared for Quiz model...")

            return {
                "success": True,
                "bullet_points": bullet_points,
                "markdown": notes_markdown,
                "metadata": {
                    "num_bullets": num_bullets,
                    "model": "ahmedhugging12/flan-t5-base-vtssum"
                }
            }

        except Exception as e:
            logger.error(f"  ✗ Notes generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _step_quiz(self, transcript: str, pdf_path: Optional[str] = None) -> Dict[str, Any]:
        """Step 6: Generate quiz questions."""
        try:
            from quiz_generator_v2 import QuizGeneratorV2
            import time

            # AGGRESSIVE MEMORY CLEANUP before loading large Quiz model
            # Quiz works in isolation but fails in pipeline - likely due to CUDA context pollution
            logger.info("  Aggressive memory cleanup before Quiz...")

            # CRITICAL: Delete large model references (especially ASR/Whisper)
            # These are the main GPU memory consumers
            if self.asr is not None:
                logger.info("  Unloading ASR model...")
                try:
                    if hasattr(self.asr, 'cleanup'):
                        self.asr.cleanup()
                    self.asr = None
                except Exception as e:
                    logger.warning(f"  ASR cleanup warning: {e}")
                    self.asr = None

            # Clear other model references
            self.engagement = None
            self.alignment = None
            self.notes_gen = None

            # Clear Python objects (run twice for thorough cleanup)
            gc.collect()
            gc.collect()

            # CUDA CONTEXT RESET - More aggressive than just empty_cache
            if torch.cuda.is_available():
                # Clear all cached memory
                torch.cuda.empty_cache()

                # Clear IPC handles
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass  # May not be available on all systems

                # Synchronize to ensure all operations complete
                torch.cuda.synchronize()

                # Reset peak memory stats (helps with fragmentation tracking)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()

                # Log memory state
                free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
                logger.info(f"  After cleanup: {free_mem:.2f}GB free GPU memory")

            # Force garbage collection again
            gc.collect()

            # Longer delay to let CUDA context stabilize
            time.sleep(5)  # Increased from 3 to 5 seconds

            logger.info("  Memory cleanup complete")

            # Check available memory before loading large model
            if torch.cuda.is_available():
                free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
                logger.info(f"  Available GPU memory: {free_mem:.2f}GB")
                if free_mem < 2.5:  # Quiz model needs ~2.1GB + buffer
                    logger.warning(f"  ⚠ Insufficient memory ({free_mem:.2f}GB < 2.5GB) - Skipping Quiz")
                    return {
                        "success": False,
                        "skipped": True,
                        "reason": f"Insufficient GPU memory ({free_mem:.2f}GB available, need 2.5GB)",
                        "questions": [],
                        "mcq_questions": [],
                        "open_ended_questions": "",
                        "num_mcq": 0,
                        "num_open_ended": 0
                    }

            logger.info("  Loading Quiz Generator (4-bit)...")
            self.quiz_gen = QuizGeneratorV2()

            try:
                self.quiz_gen.load_model()
            except (RuntimeError, OSError) as e:
                # Catch OOM and paging file errors during model loading
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "cuda" in error_msg or "paging file" in error_msg:
                    logger.error(f"  ✗ Quiz model loading failed (Memory): {e}")
                    self.quiz_gen = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Determine error type
                    if "paging file" in error_msg:
                        reason = f"System paging file too small: {str(e)[:200]}"
                    else:
                        reason = f"GPU out of memory during model loading: {str(e)[:200]}"

                    return {
                        "success": False,
                        "skipped": True,
                        "reason": reason,
                        "questions": [],
                        "mcq_questions": [],
                        "open_ended_questions": "",
                        "num_mcq": 0,
                        "num_open_ended": 0
                    }
                else:
                    raise  # Re-raise if not memory-related error

            # Load knowledge base if PDF provided
            if pdf_path and Path(pdf_path).exists():
                logger.info(f"  Loading knowledge base from PDF...")
                self.quiz_gen.load_knowledge_base([pdf_path])

            logger.info("  Generating MCQ questions...")
            mcq_questions = self.quiz_gen.generate_mcq_questions(
                context=transcript,
                num_questions=5,
                max_new_tokens=800  # ~160 tokens per question
            )

            logger.info(f"  ✓ Generated {len(mcq_questions)} MCQ questions")

            # Generate open-ended questions with sample answers (short answer type)
            logger.info("  Generating open-ended questions with sample answers...")
            open_ended_text = self.quiz_gen.generate_open_ended_questions(
                context=transcript,
                num_questions=3,
                question_type="short_answer",
                include_answers=True,  # Generate sample answers for grading guidance
                max_new_tokens=1000  # More tokens to accommodate answers
            )

            logger.info(f"  ✓ Generated open-ended questions")

            # Unload quiz generator
            self._unload_model("quiz")

            return {
                "success": True,
                "mcq_questions": mcq_questions,
                "open_ended_questions": open_ended_text,
                "num_mcq": len(mcq_questions),
                "num_open_ended": 3,
                # Keep 'questions' for backward compatibility
                "questions": mcq_questions
            }

        except RuntimeError as e:
            # Catch any remaining OOM errors
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                logger.error(f"  ✗ Quiz generation failed (OOM): {e}")
                if self.quiz_gen:
                    self.quiz_gen = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {
                    "success": False,
                    "skipped": True,
                    "reason": f"GPU out of memory: {str(e)[:200]}",
                    "questions": [],
                    "mcq_questions": [],
                    "open_ended_questions": "",
                    "num_mcq": 0,
                    "num_open_ended": 0
                }
            else:
                logger.error(f"  ✗ Quiz generation failed: {e}")
                return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"  ✗ Quiz generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary and combined scores."""
        summary = {
            "lecture_title": results.get("lecture_title", "Unknown"),
            "processing_complete": True,
            "steps_completed": results.get("processing_steps", [])
        }

        # Extract key metrics
        if "engagement" in results and results["engagement"].get("success"):
            summary["engagement_score"] = results["engagement"]["engagement_score"]

        if "content_alignment" in results and results["content_alignment"].get("success"):
            summary["coverage_score"] = results["content_alignment"]["coverage_score"]

        # Combined score (if both available)
        if "engagement_score" in summary and "coverage_score" in summary:
            # Simple average for now (can be weighted)
            combined = (summary["engagement_score"] + summary["coverage_score"]) / 2
            summary["combined_score"] = round(combined, 2)
            summary["grade"] = self._score_to_grade(combined)

        # Content generated
        summary["outputs"] = {
            "transcript_generated": results.get("transcript", {}).get("success", False),
            "notes_generated": results.get("notes", {}).get("success", False),
            "quiz_generated": results.get("quiz", {}).get("success", False)
        }

        return summary

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Very Good)"
        elif score >= 70:
            return "C (Good)"
        elif score >= 60:
            return "D (Fair)"
        else:
            return "F (Needs Improvement)"

    def _needs_translation(self, text: str) -> bool:
        """Check if text contains Arabic and needs translation."""
        import re
        has_arabic = bool(re.search(r'[\u0600-\u06FF]', text))
        return has_arabic

    def _unload_model(self, model_name: str):
        """Unload a model and free GPU memory."""
        try:
            if model_name == "asr":
                self.asr = None
            elif model_name == "alignment":
                self.alignment = None
            elif model_name == "notes":
                self.notes_gen = None
            elif model_name == "quiz":
                self.quiz_gen = None

            # Force garbage collection
            gc.collect()

            # Clear GPU cache if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                logger.info(f"  Memory after unload: {allocated:.2f}GB")

        except Exception as e:
            logger.warning(f"  Warning during unload: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the teacher module."""
        status = {
            "models_loaded": {
                "asr": self.asr is not None,
                "engagement": self.engagement is not None,
                "alignment": self.alignment is not None,
                "translator": self.translator is not None,
                "notes": self.notes_gen is not None,
                "quiz": self.quiz_gen is not None
            },
            "options": {
                "use_punctuation": self.use_punctuation,
                "use_translation": self.use_translation,
                "enable_quiz": self.enable_quiz,
                "enable_notes": self.enable_notes
            }
        }

        if TORCH_AVAILABLE and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            status["gpu_memory"] = {
                "allocated": round(allocated, 2),
                "total": round(total, 2),
                "utilization_percent": round(100 * allocated / total, 1)
            }

        return status


# Factory function
def create_teacher_module(**kwargs) -> TeacherModuleV2:
    """Create and configure teacher module."""
    return TeacherModuleV2(**kwargs)


# Test function
if __name__ == "__main__":
    logger.info("Testing TeacherModule V2")

    # Create module
    teacher = create_teacher_module(
        use_punctuation=False,  # Skip punctuation (very slow)
        use_translation=True,
        enable_quiz=True,
        enable_notes=True
    )

    # Print status
    status = teacher.get_status()
    logger.info(f"Module Status: {status}")

    logger.info("\nTeacherModule ready for testing!")
    logger.info("Use test_teacher_module_e2e.py for full pipeline testing")
