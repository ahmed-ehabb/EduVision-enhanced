"""
Text alignment module for classroom AI system.
This module analyzes lecture transcripts to determine alignment with textbook content.

Implementation matches the Colab notebook analysis approach exactly:
- Uses paraphrase-MiniLM-L6-v2 SBERT model for embeddings
- Similarity thresholds: Fully Covered ≥0.75, Partially Covered ≥0.5, Off-topic <0.5
- Weighted scoring: Fully Covered=1.0, Partially Covered=0.5, Off-topic=0.0
- Exact paragraph cleaning and segmentation from notebook
- Exact feedback messages based on final score ranges
"""

import re
import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from typing import List, Dict, Tuple, Optional
import fitz
# Import for PDF processing
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Warning: PyMuPDF not installed. PDF processing will not be available.")
    print("Install with: pip install pymupdf")

# Sentence transformers for semantic matching
try:
    from sentence_transformers import SentenceTransformer, util

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not installed. Using fallback matching.")
    print("Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TextAlignmentAnalyzer:
    """Analyzes alignment between lecture transcripts and textbook content."""

    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        """Initialize the analyzer with the specified embedding model."""
        self.textbook_paragraphs = []
        self.textbook_embeddings = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load embedding model if available - exactly as in notebook
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"[GEAR] Loading text alignment model ({model_name})...")
                self.model = SentenceTransformer(model_name)
                self.model.to(self.device)
                print(f"[OK] Text alignment model loaded on {self.device}")
            except Exception as e:
                print(f"[ERROR] Error loading sentence transformer model: {e}")
                self.model = None

    def load_textbook_from_pdf(
        self,
        pdf_path: str,
        chapter_start_anchor: Optional[str] = None,
        chapter_end_anchor: Optional[str] = None,
    ) -> List[str]:
        """
        Extract and clean paragraphs from a textbook PDF - matches notebook approach exactly.

        Args:
            pdf_path: Path to the PDF file
            chapter_start_anchor: Text to identify start of extraction (e.g., notebook uses "Psychology is the scientific study of mind and behavior")
            chapter_end_anchor: Text to identify end of extraction (e.g., notebook uses "Biopsychology is the study of how biology influences behavior")

        Returns:
            List of cleaned textbook paragraphs
        """
        if not os.path.exists(pdf_path):
            print(f"[ERROR] Error: PDF file not found at {pdf_path}")
            return []

        try:
            print(f"📥 Loading textbook from PDF: {pdf_path}")
            doc = fitz.open(pdf_path)

            # Extract text based on anchors - exactly as in notebook
            capture = (
                chapter_start_anchor is None
            )  # Start capturing immediately if no anchor
            full_text = ""
            chapter_1_found = False

            for page in doc:
                text = page.get_text()

                if (
                    chapter_start_anchor
                    and chapter_start_anchor in text
                    and not chapter_1_found
                ):
                    chapter_1_found = True
                    capture = True

                if chapter_end_anchor and chapter_end_anchor in text:
                    break

                if capture:
                    full_text += text + "\n"

            # Clean and segment the text into paragraphs using exact notebook approach
            paragraphs = self._clean_and_segment_paragraphs(full_text)
            self.textbook_paragraphs = paragraphs

            # Generate embeddings if model is available
            if self.model:
                print("[GEAR] Generating textbook paragraph embeddings...")
                self.textbook_embeddings = self.model.encode(
                    paragraphs, convert_to_tensor=True, device=self.device
                )
                print(f"[OK] Processed {len(paragraphs)} textbook paragraphs from PDF")

            print("[OK] Textbook loaded successfully!")
            return paragraphs

        except Exception as e:
            print(f"[ERROR] Error processing PDF: {e}")
            return []

    def load_textbook_from_text(self, text: str) -> List[str]:
        """
        Load textbook content from a text string instead of PDF.

        Args:
            text: The textbook content as a string

        Returns:
            List of cleaned textbook paragraphs
        """
        paragraphs = self._clean_and_segment_paragraphs(text)
        self.textbook_paragraphs = paragraphs

        # Generate embeddings if model is available
        if self.model:
            print("[GEAR] Generating textbook paragraph embeddings...")
            self.textbook_embeddings = self.model.encode(
                paragraphs, convert_to_tensor=True, device=self.device
            )

        print(f"[OK] Processed {len(paragraphs)} textbook paragraphs from text")
        return paragraphs

    def _clean_and_segment_paragraphs(self, text: str) -> List[str]:
        """
        Clean and segment text into paragraphs - exactly as in notebook.

        Args:
            text: Raw text to clean and segment

        Returns:
            List of cleaned paragraphs
        """
        # Remove unwanted headers, footers, page numbers - exactly as notebook
        text = re.sub(r"Page \d+", "", text)
        text = re.sub(r"OpenStax.*?\n", "", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)

        # Remove non-educational section titles - exactly as notebook
        text = re.sub(
            r"(Summary|Key Terms|Review Questions|Critical Thinking Questions|Personal Application Questions).*?",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Split by sentence boundaries - exactly as notebook
        paragraphs = re.split(r"(?<=[.?!])\s+(?=[A-Z])", text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]

        # Final filter: remove figure labels, credits, outlines, and overly short entries - exactly as notebook
        filtered_paragraphs = [
            p
            for p in paragraphs
            if not p.lower().startswith("figure")
            and "credit" not in p.lower()
            and "modification of work" not in p.lower()
            and "chapter outline" not in p.lower()
            and len(p.strip()) > 80
        ]

        return filtered_paragraphs

    def process_transcript(self, transcript: str) -> List[str]:
        """
        Process a lecture transcript into segments for analysis - exactly as notebook.

        Args:
            transcript: The lecture transcript text

        Returns:
            List of transcript segments
        """
        # Basic cleaning - exactly as notebook
        cleaned_transcript = re.sub(r"\n+", "\n", transcript)
        cleaned_transcript = re.sub(r"[ \t]+", " ", cleaned_transcript)
        cleaned_transcript = cleaned_transcript.strip()

        # Segment into chunks using sentence and paragraph boundaries - exactly as notebook
        segments = re.split(r"(?<=[.?!])\s+(?=[A-Z])|\n{2,}", cleaned_transcript)
        segments = [seg.strip() for seg in segments if len(seg.strip()) > 40]

        # Filter out very short segments - exactly as notebook
        final_segments = [s for s in segments if len(s.split()) >= 6]

        return final_segments

    def analyze_alignment(
        self,
        transcript_segments: List[str],
        similarity_thresholds: Dict[str, float] = None,
    ) -> Dict:
        """
        Analyze alignment between transcript segments and textbook paragraphs - exactly as notebook.

        Args:
            transcript_segments: List of transcript segments to analyze
            similarity_thresholds: Dict with thresholds for coverage categories

        Returns:
            Dict with analysis results matching notebook format
        """
        if not self.textbook_paragraphs:
            return {"error": "No textbook content loaded"}

        if not self.model:
            return {"error": "No embedding model available"}

        # Use exact thresholds from notebook
        if similarity_thresholds is None:
            similarity_thresholds = {
                "fully_covered": 0.75,  # Exactly as notebook
                "partially_covered": 0.5,  # Exactly as notebook
            }

        # Generate embeddings for transcript segments
        print("[GEAR] Generating transcript segment embeddings...")
        transcript_embeddings = self.model.encode(
            transcript_segments, convert_to_tensor=True, device=self.device
        )

        # Initialize results list
        coverage_results = []

        print(
            f"[CHART] Comparing {len(transcript_segments)} transcript segments to {len(self.textbook_paragraphs)} textbook paragraphs..."
        )

        # Compare each transcript segment to all textbook paragraphs - exactly as notebook
        for i, t_embed in enumerate(transcript_embeddings):
            # Compute cosine similarity with all textbook embeddings
            similarity_scores = util.cos_sim(t_embed, self.textbook_embeddings)[0]

            # Get the best match (highest similarity)
            best_score = similarity_scores.max().item()
            best_match_index = similarity_scores.argmax().item()

            # Get the matching paragraph text
            best_match_paragraph = self.textbook_paragraphs[best_match_index]

            # Classify coverage based on exact notebook thresholds
            if best_score >= similarity_thresholds["fully_covered"]:  # ≥0.75
                coverage = "Fully Covered"
            elif best_score >= similarity_thresholds["partially_covered"]:  # ≥0.5
                coverage = "Partially Covered"
            else:
                coverage = "Off-topic"

            # Store the result - exactly as notebook format
            coverage_results.append(
                {
                    "Transcript Segment": transcript_segments[i],
                    "Matched Textbook Paragraph": best_match_paragraph,
                    "Similarity Score": round(best_score, 4),
                    "Coverage Label": coverage,
                }
            )

        # Calculate coverage statistics - exactly as notebook
        coverage_df = pd.DataFrame(coverage_results)
        coverage_counts = coverage_df["Coverage Label"].value_counts()
        total_segments = len(coverage_df)

        # Calculate percentages for each category - exactly as notebook
        coverage_percentages = {}
        for category in ["Fully Covered", "Partially Covered", "Off-topic"]:
            count = coverage_counts.get(category, 0)
            percentage = (count / total_segments * 100) if total_segments > 0 else 0
            coverage_percentages[category] = round(percentage, 2)

        # Calculate final score using exact notebook weighted approach
        weight_fully = 1.0  # Exactly as notebook
        weight_partial = 0.5  # Exactly as notebook
        weight_offtopic = 0.0  # Exactly as notebook

        final_score = (
            coverage_percentages.get("Fully Covered", 0) * weight_fully
            + coverage_percentages.get("Partially Covered", 0) * weight_partial
            + coverage_percentages.get("Off-topic", 0) * weight_offtopic
        )

        # Generate feedback based on exact notebook score ranges and messages
        if final_score >= 85:
            feedback = "Outstanding job — the lecture was highly aligned with the textbook and maintained excellent coverage."
        elif final_score >= 70:
            feedback = "Very good — the lecture was well-aligned with the syllabus, with some room to improve."
        elif final_score >= 50:
            feedback = "Fair — but significant portions of the lecture did not fully reflect textbook material."
        else:
            feedback = "Needs improvement — much of the lecture content diverged from textbook objectives."

        print(f"[BLUE_BOOK] Coverage analysis completed: {final_score:.2f}% alignment score")

        # Compile the results - exactly as notebook format with additional fields for API compatibility
        results = {
            "status": "completed",
            "textbook_reference": "Psychology2e_WEB.pdf",
            "coverage_results": coverage_results,
            "coverage_counts": {k: int(v) for k, v in coverage_counts.items()},
            "coverage_percentages": coverage_percentages,
            "final_score": round(final_score, 2),
            "alignment_score": round(final_score / 100, 4),  # For API compatibility (decimal format)
            "feedback": feedback,
            "total_segments": total_segments,
            "similarity_thresholds": similarity_thresholds,
            "textbook_sections": len(self.textbook_paragraphs),
            
            # Additional fields for dashboard display
            "coverage_analysis": {
                "fully_covered": coverage_percentages.get("Fully Covered", 0),
                "partially_covered": coverage_percentages.get("Partially Covered", 0),
                "off_topic": coverage_percentages.get("Off-topic", 0)
            },
            "similarity_threshold": {
                "fully_covered": similarity_thresholds["fully_covered"],
                "partially_covered": similarity_thresholds["partially_covered"]
            },
            "processing_method": "SBERT (paraphrase-MiniLM-L6-v2)",
            "analysis_method": "SBERT + Psychology Textbook",
            "cache_used": False
        }

        return results

    def generate_visualization_data(self, results: Dict) -> Dict:
        """
        Generate data needed for visualizations - matches notebook output.

        Args:
            results: The analysis results from analyze_alignment

        Returns:
            Dict with visualization data matching notebook format
        """
        if "coverage_results" not in results:
            return {}

        # Prepare coverage distribution for pie chart - exactly as notebook
        coverage_percentages = results["coverage_percentages"]
        coverage_data = []

        # Ensure we have all three categories in the correct order
        for category in ["Fully Covered", "Partially Covered", "Off-topic"]:
            percentage = coverage_percentages.get(category, 0.0)
            coverage_data.append(
                {
                    "category": category,
                    "percentage": percentage,
                    "count": results["coverage_counts"].get(category, 0),
                }
            )

        # Prepare data for visualization matching notebook
        visualization_data = {
            "coverage_distribution": coverage_data,
            "final_score": results["final_score"],
            "feedback": results["feedback"],
            "total_segments": results["total_segments"],
        }

        return visualization_data

    def analyze_text(
        self, text: str, similarity_thresholds: Dict[str, float] = None
    ) -> Dict:
        """
        Complete text alignment analysis workflow - processes text and analyzes against default textbook.

        Args:
            text: Input text (transcript) to analyze
            similarity_thresholds: Dict with thresholds for coverage categories (uses notebook defaults if None)

        Returns:
            Dict with complete analysis results matching notebook format
        """
        try:
            # Load default textbook content if not already loaded
            if not self.textbook_paragraphs:
                self._load_default_textbook()

            # Process the input text into segments using notebook approach
            transcript_segments = self.process_transcript(text)

            if not transcript_segments:
                return {
                    "status": "error",
                    "error": "No valid segments found in input text",
                    "textbook_reference": "Psychology2e_WEB.pdf",
                    "coverage_results": [],
                    "coverage_counts": {},
                    "coverage_percentages": {
                        "Fully Covered": 0.0,
                        "Partially Covered": 0.0,
                        "Off-topic": 0.0,
                    },
                    "coverage_analysis": {
                        "fully_covered": 0.0,
                        "partially_covered": 0.0,
                        "off_topic": 0.0
                    },
                    "final_score": 0.0,
                    "alignment_score": 0.0,
                    "feedback": "No analyzable content found in the input text.",
                    "total_segments": 0,
                    "textbook_sections": len(self.textbook_paragraphs) if self.textbook_paragraphs else 0,
                    "analysis_method": "SBERT (paraphrase-MiniLM-L6-v2) + Psychology Textbook",
                    "processing_method": "SBERT (paraphrase-MiniLM-L6-v2)",
                    "similarity_threshold": {
                        "fully_covered": 0.75,
                        "partially_covered": 0.5
                    },
                    "cache_used": False
                }

            # Perform the alignment analysis using exact notebook approach
            return self.analyze_alignment(transcript_segments, similarity_thresholds)

        except Exception as e:
            return {
                "status": "error",
                "error": f"Text alignment analysis failed: {str(e)}",
                "textbook_reference": "Psychology2e_WEB.pdf",
                "coverage_results": [],
                "coverage_counts": {},
                "coverage_percentages": {
                    "Fully Covered": 0.0,
                    "Partially Covered": 0.0,
                    "Off-topic": 0.0,
                },
                "coverage_analysis": {
                    "fully_covered": 0.0,
                    "partially_covered": 0.0,
                    "off_topic": 0.0
                },
                "final_score": 0.0,
                "alignment_score": 0.0,
                "feedback": f"Analysis failed due to error: {str(e)}",
                "total_segments": 0,
                "textbook_sections": 0,
                "analysis_method": "SBERT (paraphrase-MiniLM-L6-v2) + Psychology Textbook",
                "processing_method": "SBERT (paraphrase-MiniLM-L6-v2)",
                "similarity_threshold": {
                    "fully_covered": 0.75,
                    "partially_covered": 0.5
                },
                "cache_used": False
            }

    def _load_default_textbook(self):
        """Load default textbook content for analysis - psychology focused as in notebook."""
        # Default textbook content focused on psychology topics (Chapters 1 & 2 content)
        default_textbook = """
        Psychology is the scientific study of mind and behavior. Psychology includes the study of conscious and unconscious phenomena, including feelings and thoughts. It is an academic discipline of immense scope, crossing the boundaries between the natural and social sciences. Psychologists seek an understanding of the emergent properties of brains, linking the discipline to neuroscience. As social scientists, psychologists aim to understand the behavior of individuals and groups.
        
        The word psychology derives from the Greek word psyche, for spirit or soul. The Latin word psychologia was first used by the Croatian humanist and Latinist Marko Marulić in his book Psichiologia de ratione animae humanae in the late 15th or early 16th century. The earliest known reference to the word psychology in English was by Steven Blankaart in 1694 in The Physical Dictionary. The dictionary refers to "Anatomy, which treats the Body, and Psychology, which treats of the Soul."
        
        In 1890, William James defined psychology as "the science of mental life, both of its phenomena and their conditions." This definition enjoyed widespread currency for decades. However, this meaning was contested, notably by radical behaviorists such as John B. Watson, who in his 1913 manifesto defined the discipline of psychology as the acquisition of information useful to the control of behavior.
        
        Wilhelm Wundt is credited with the establishment of psychology as an independent empirical science through his creation of the first laboratory dedicated exclusively to psychological research in Leipzig in 1879. Wundt is known for his use of introspection as a method for psychological research, though introspection was not unique to Wundt.
        
        Behaviorism dominated experimental psychology for several decades, and influenced thinking in many other disciplines. Pavlov demonstrated conditioning with dogs and sparked off a revolution in the study of psychology. From the 1950s, the experimental science of behaviorism came under fire for its inability to adequately explain language and cognition. It was gradually replaced by cognitive psychology, which uses an information-processing model of mental function.
        
        Cognitive psychology is the scientific study of mental processes such as attention, language use, memory, perception, problem solving, creativity, and reasoning. Cognitive psychology originated in the 1960s in a break from behaviorism, which had held from the 1920s to 1950s that unobservable mental processes were outside of the realm of empirical science.
        
        Research methods in psychology include experiments, observations, interviews, and psychological testing. Experimental methods involve the manipulation of variables in order to determine cause and effect relationships. The experimental method involves manipulating one variable to determine if changes in one variable cause changes in another variable.
        
        Psychological research strives to understand and explain how people think, act, and feel. Psychology research can usually be classified as basic or applied. Basic research in psychology is conducted primarily for the sake of increasing our knowledge base and is typically conducted in universities and research institutions.
        
        Ethics in psychological research is critically important. The American Psychological Association (APA) has developed a set of ethical principles and standards that psychologists must follow when conducting research with human participants. These include informed consent, confidentiality, and the minimization of harm.
        
        Biopsychology explores how our biology influences our behavior. While biological psychology is a broad field, many biological psychologists want to understand how the structure and function of the nervous system is related to behavior. As such, they often combine the research strategies of both psychologists and physiologists to accomplish this goal.
        """

        # Process using the same approach as PDF loading
        paragraphs = self._clean_and_segment_paragraphs(default_textbook)
        self.textbook_paragraphs = paragraphs

        # Generate embeddings if model is available
        if self.model:
            print("[GEAR] Generating default textbook embeddings...")
            self.textbook_embeddings = self.model.encode(
                paragraphs, convert_to_tensor=True, device=self.device
            )
            print(f"[OK] Loaded {len(paragraphs)} default textbook paragraphs")


def get_text_alignment_analyzer():
    """Factory function to get a TextAlignmentAnalyzer instance."""
    return TextAlignmentAnalyzer()
