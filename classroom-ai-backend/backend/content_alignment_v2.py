"""
Content Alignment Module V2 - Text Alignment Using SBERT

Analyzes lecture content alignment with textbook material using:
- SBERT (Sentence-BERT) embeddings
- Cosine similarity between transcript and textbook
- Coverage classification (Fully Covered, Partially Covered, Off-topic)

Model: sentence-transformers/paraphrase-MiniLM-L6-v2
Processing: CPU or GPU (automatic detection)
Memory: Minimal (~200MB model)

Author: Based on the content alignment notebook
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


class ContentAlignmentAnalyzer:
    """
    Analyzes lecture content alignment with textbook material using SBERT.

    Uses cosine similarity to match transcript segments with textbook paragraphs.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        fully_covered_threshold: float = 0.60,
        partially_covered_threshold: float = 0.40
    ):
        """
        Initialize the content alignment analyzer.

        Args:
            model_name: SBERT model name (default: paraphrase-MiniLM-L6-v2)
            fully_covered_threshold: Similarity threshold for "Fully Covered" (default: 0.60, was 0.75)
            partially_covered_threshold: Similarity threshold for "Partially Covered" (default: 0.40, was 0.50)
        """
        if not SBERT_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

        self.model_name = model_name
        self.fully_threshold = fully_covered_threshold
        self.partially_threshold = partially_covered_threshold
        self.model = None
        self.textbook_embeddings = None
        self.textbook_paragraphs = None

        logger.info(f"[ContentAlignment] Initializing with model: {model_name}")

    def load_model(self):
        """Load the SBERT model."""
        if self.model is None:
            logger.info(f"[ContentAlignment] Loading SBERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"[ContentAlignment] Model loaded successfully")

    def load_textbook_content(self, textbook_paragraphs: List[str]):
        """
        Load and embed textbook content.

        Args:
            textbook_paragraphs: List of textbook paragraph strings
        """
        self.load_model()

        self.textbook_paragraphs = textbook_paragraphs

        logger.info(f"[ContentAlignment] Encoding {len(textbook_paragraphs)} textbook paragraphs...")
        self.textbook_embeddings = self.model.encode(
            textbook_paragraphs,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        logger.info(f"[ContentAlignment] Textbook embeddings created")

    def analyze_transcript(
        self,
        transcript_segments: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze transcript alignment with textbook content.

        Args:
            transcript_segments: List of transcript segment strings

        Returns:
            Dictionary containing alignment results and statistics
        """
        if self.textbook_embeddings is None:
            raise ValueError("Textbook content not loaded. Call load_textbook_content() first.")

        self.load_model()

        # Encode transcript segments
        logger.info(f"[ContentAlignment] Encoding {len(transcript_segments)} transcript segments...")
        transcript_embeddings = self.model.encode(
            transcript_segments,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # Analyze each segment
        results = []

        for i, t_embed in enumerate(transcript_embeddings):
            # Compute cosine similarity with all textbook paragraphs
            similarity_scores = util.cos_sim(t_embed, self.textbook_embeddings)[0]

            # Get best match
            best_score = float(similarity_scores.max().item())
            best_match_idx = int(similarity_scores.argmax().item())
            best_match_paragraph = self.textbook_paragraphs[best_match_idx]

            # Classify coverage
            if best_score >= self.fully_threshold:
                coverage_label = "Fully Covered"
            elif best_score >= self.partially_threshold:
                coverage_label = "Partially Covered"
            else:
                coverage_label = "Off-topic"

            results.append({
                "segment_id": i,
                "transcript_segment": transcript_segments[i],
                "matched_textbook_paragraph": best_match_paragraph,
                "similarity_score": round(best_score, 4),
                "coverage_label": coverage_label
            })

        # Calculate statistics
        df_results = pd.DataFrame(results)

        # Coverage counts
        coverage_counts = df_results["coverage_label"].value_counts()
        total_segments = len(df_results)

        # Coverage percentages
        coverage_percentages = {}
        for label in ["Fully Covered", "Partially Covered", "Off-topic"]:
            count = coverage_counts.get(label, 0)
            coverage_percentages[label] = {
                "count": int(count),
                "percentage": round(100 * count / total_segments, 2)
            }

        # Overall coverage score (weighted)
        coverage_weights = {
            "Fully Covered": 1.0,
            "Partially Covered": 0.5,
            "Off-topic": 0.0
        }

        coverage_score = 100 * df_results["coverage_label"].map(coverage_weights).mean()

        # Calculate which textbook topics/paragraphs were covered
        # A paragraph is considered "covered" if at least one transcript segment matched it
        covered_paragraph_indices = set()
        for result in results:
            if result["coverage_label"] in ["Fully Covered", "Partially Covered"]:
                # Find which paragraph index was matched
                matched_para = result["matched_textbook_paragraph"]
                for idx, para in enumerate(self.textbook_paragraphs):
                    if para == matched_para:
                        covered_paragraph_indices.add(idx)
                        break

        num_total_topics = len(self.textbook_paragraphs)
        num_covered_topics = len(covered_paragraph_indices)
        coverage_percentage = round(100 * num_covered_topics / num_total_topics, 2) if num_total_topics > 0 else 0

        # Get list of covered topic texts (first 50 chars of each paragraph as topic name)
        covered_topics = [
            self.textbook_paragraphs[idx][:50] + ("..." if len(self.textbook_paragraphs[idx]) > 50 else "")
            for idx in sorted(covered_paragraph_indices)
        ]

        # Generate feedback
        feedback = self._generate_feedback(coverage_score)

        logger.info(f"[ContentAlignment] Analysis complete. Coverage score: {coverage_score:.2f}%")
        logger.info(f"[ContentAlignment] Topics covered: {num_covered_topics}/{num_total_topics} ({coverage_percentage}%)")

        return {
            "results": results,
            "coverage_score": round(coverage_score, 2),
            "coverage_percentages": coverage_percentages,
            "coverage_percentage": coverage_percentage,
            "num_covered_topics": num_covered_topics,
            "num_total_topics": num_total_topics,
            "covered_topics": covered_topics,
            "feedback": feedback,
            "total_segments": total_segments,
            "model_used": self.model_name
        }

    def _generate_feedback(self, coverage_score: float) -> str:
        """Generate feedback message based on coverage score."""
        if coverage_score >= 85:
            return "Outstanding job — the lecture was highly aligned with the textbook and maintained excellent coverage."
        elif coverage_score >= 70:
            return "Very good — the lecture was well-aligned with the syllabus, with some room to improve."
        elif coverage_score >= 50:
            return "Fair — but significant portions of the lecture did not fully reflect textbook material."
        else:
            return "Needs improvement — much of the lecture content diverged from textbook objectives."

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration info."""
        return {
            "model_name": self.model_name,
            "model_type": "sentence-transformers (SBERT)",
            "fully_covered_threshold": self.fully_threshold,
            "partially_covered_threshold": self.partially_threshold,
            "gpu_required": False,
            "memory_usage": "minimal (~200MB)"
        }


# Helper function for extracting textbook content (simplified version)
def extract_textbook_paragraphs(text: str, min_length: int = 80) -> List[str]:
    """
    Extract and clean paragraphs from textbook text.

    Args:
        text: Raw textbook text
        min_length: Minimum paragraph length in characters

    Returns:
        List of cleaned paragraphs
    """
    # Remove headers, footers, page numbers
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'OpenStax.*?\n', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Remove non-educational sections
    text = re.sub(
        r'(Summary|Key Terms|Review Questions|Critical Thinking Questions|Personal Application Questions).*?',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Split by sentence boundaries
    paragraphs = re.split(r'(?<=[.?!])\s+(?=[A-Z])', text)
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > min_length]

    # Filter out figure labels, credits, etc.
    filtered_paragraphs = [
        p for p in paragraphs
        if not p.lower().startswith("figure")
        and "credit" not in p.lower()
        and "modification of work" not in p.lower()
        and "chapter outline" not in p.lower()
    ]

    return filtered_paragraphs


# Helper function for segmenting transcript
def segment_transcript(transcript: str, min_words: int = 6) -> List[str]:
    """
    Segment transcript into meaningful chunks.

    Args:
        transcript: Raw transcript text
        min_words: Minimum words per segment

    Returns:
        List of transcript segments
    """
    # Clean transcript
    cleaned = re.sub(r'\n+', '\n', transcript)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = cleaned.strip()

    # Split by sentence/paragraph boundaries
    segments = re.split(r'(?<=[.?!])\s+(?=[A-Z])|\n{2,}', cleaned)
    segments = [seg.strip() for seg in segments if len(seg.strip()) > 40]

    # Filter short segments
    final_segments = [s for s in segments if len(s.split()) >= min_words]

    return final_segments


# Factory function
def create_content_alignment_analyzer() -> ContentAlignmentAnalyzer:
    """
    Create and configure content alignment analyzer.

    Returns:
        Initialized ContentAlignmentAnalyzer instance
    """
    return ContentAlignmentAnalyzer()


# Test function
def test_content_alignment():
    """Test content alignment analyzer."""
    logger.info("="*80)
    logger.info("Testing Content Alignment Analyzer")
    logger.info("="*80)

    if not SBERT_AVAILABLE:
        logger.error("sentence-transformers not available")
        return

    # Sample textbook content
    textbook_paragraphs = [
        "Psychology is the scientific study of mind and behavior. It encompasses the study of conscious and unconscious phenomena.",
        "The scientific method involves forming hypotheses, conducting experiments, and analyzing data systematically.",
        "Operant conditioning is a learning process where behavior is modified by consequences such as reinforcement or punishment.",
        "Classical conditioning involves learning through association, as demonstrated by Pavlov's experiments with dogs.",
        "The brain consists of billions of neurons that communicate through electrical and chemical signals."
    ]

    # Sample transcript
    transcript_segments = [
        "Today we're going to talk about what psychology really is. It's the scientific study of how we think and behave.",
        "Psychologists use the scientific method to test their ideas. They form hypotheses and run experiments.",
        "Remember Pavlov's dogs? That's a classic example of classical conditioning where learning happens through association.",
        "The weather today is really nice, I hope you all had a good breakfast this morning.",
        "Operant conditioning is different - it's about learning from consequences like rewards and punishments."
    ]

    # Create analyzer
    analyzer = create_content_alignment_analyzer()

    # Print model info
    info = analyzer.get_model_info()
    logger.info(f"\nModel Info: {info}")

    # Load textbook
    logger.info(f"\nLoading {len(textbook_paragraphs)} textbook paragraphs...")
    analyzer.load_textbook_content(textbook_paragraphs)

    # Analyze transcript
    logger.info(f"\nAnalyzing {len(transcript_segments)} transcript segments...")
    result = analyzer.analyze_transcript(transcript_segments)

    # Print results
    logger.info("\n" + "="*80)
    logger.info("Analysis Results")
    logger.info("="*80)

    logger.info(f"\nOverall Coverage Score: {result['coverage_score']:.2f}%")
    logger.info(f"Feedback: {result['feedback']}")

    logger.info(f"\nCoverage Breakdown:")
    for label, stats in result['coverage_percentages'].items():
        logger.info(f"  {label}: {stats['count']} segments ({stats['percentage']:.1f}%)")

    logger.info(f"\nSegment Details:")
    for r in result['results']:
        logger.info(f"\n  Segment {r['segment_id'] + 1}: {r['coverage_label']} (similarity: {r['similarity_score']:.3f})")
        logger.info(f"    Transcript: {r['transcript_segment'][:80]}...")
        logger.info(f"    Best match: {r['matched_textbook_paragraph'][:80]}...")

    logger.info("\n" + "="*80)
    logger.info("Test Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    test_content_alignment()
