"""
Engagement Analyzer V2 - Audio-Only Implementation

Analyzes teacher/speaker engagement from audio using:
- Loudness (RMS energy) - 60% weight
- Pitch variation (F0 std) - 40% weight

Optimized for RTX 3050 - runs entirely on CPU (no GPU needed)

Author: Based on the audio engagement analysis from the graduation project
"""

import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Tuple, Any
from pathlib import Path
import logging

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class EngagementAnalyzerV2:
    """
    Audio-based engagement analyzer using loudness and pitch variation.

    No GPU required - runs entirely on CPU using librosa.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = None,
        loudness_weight: float = 0.6,
        pitch_weight: float = 0.4
    ):
        """
        Initialize engagement analyzer.

        Args:
            sample_rate: Target sample rate for audio (16kHz default)
            hop_length: Hop length for feature extraction (default: 0.025 * sr)
            loudness_weight: Weight for loudness in engagement score (default: 0.6)
            pitch_weight: Weight for pitch variation in engagement score (default: 0.4)
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length or int(0.025 * sample_rate)
        self.loudness_weight = loudness_weight
        self.pitch_weight = pitch_weight

        logger.info(f"[EngagementV2] Initialized with sr={sample_rate}, hop={self.hop_length}")

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """
        Load audio file and convert to 16kHz mono.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_data, duration_seconds)
        """
        try:
            # Load and convert to target sample rate
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            duration = len(y) / sr

            logger.info(f"[EngagementV2] Audio loaded: {duration/60:.1f} min @ {sr} Hz")
            return y, duration

        except Exception as e:
            logger.error(f"[EngagementV2] Failed to load audio: {e}")
            raise

    def create_segment_bounds(
        self,
        transcript_segments: List[str],
        duration: float
    ) -> List[Tuple[float, float]]:
        """
        Create segment time bounds based on character share.

        Args:
            transcript_segments: List of transcript text segments
            duration: Total audio duration in seconds

        Returns:
            List of (start_time, end_time) tuples
        """
        total_chars = sum(len(s) for s in transcript_segments)

        if total_chars == 0:
            logger.warning("[EngagementV2] No transcript segments, using default 30s segments")
            return self._create_default_segments(duration)

        cum = 0
        segment_bounds = []

        for seg in transcript_segments:
            start = (cum / total_chars) * duration
            cum += len(seg)
            end = (cum / total_chars) * duration
            segment_bounds.append((start, end))

        logger.info(f"[EngagementV2] Created {len(segment_bounds)} segments by char-share")
        return segment_bounds

    def _create_default_segments(
        self,
        duration: float,
        segment_length: float = 30.0
    ) -> List[Tuple[float, float]]:
        """Create fixed-length segments if no transcript provided."""
        segments = []
        current_time = 0.0

        while current_time < duration:
            end_time = min(current_time + segment_length, duration)
            segments.append((current_time, end_time))
            current_time = end_time

        return segments

    def extract_segment_features(
        self,
        audio: np.ndarray,
        segment_bounds: List[Tuple[float, float]]
    ) -> Tuple[List[float], List[float]]:
        """
        Extract loudness and pitch variation for each segment (OPTIMIZED).

        Args:
            audio: Audio signal
            segment_bounds: List of (start, end) time bounds

        Returns:
            Tuple of (loudness_values, pitch_variation_values)
        """
        import time
        start_time = time.time()

        # OPTIMIZATION: Extract pitch once for entire audio, then segment
        # Using yin() instead of pyin() for much faster processing (5-10x faster)
        logger.info(f"[EngagementV2] Extracting pitch for entire audio (optimized with yin)...")
        try:
            # Use larger hop_length for faster processing
            fast_hop = self.hop_length * 4  # 4x faster
            # Use yin() instead of pyin() - much faster, still good quality
            f0_full = librosa.yin(
                audio,
                fmin=librosa.note_to_hz('C2'),  # ~80 Hz
                fmax=librosa.note_to_hz('C7'),  # ~400 Hz
                sr=self.sample_rate,
                hop_length=fast_hop,
                frame_length=fast_hop * 4  # Larger frame for stability
            )
        except Exception as e:
            logger.warning(f"[EngagementV2] Full audio pitch extraction failed: {e}")
            f0_full = None

        loudness_vals = []
        pitch_vals = []

        for i, (start, end) in enumerate(segment_bounds):
            # Extract segment
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            segment = audio[start_sample:end_sample]

            if len(segment) == 0:
                loudness_vals.append(0.0)
                pitch_vals.append(0.0)
                continue

            # 1. Loudness = mean RMS energy (fast)
            rms = librosa.feature.rms(y=segment, hop_length=self.hop_length)[0]
            loudness = float(rms.mean())

            # 2. Pitch variation from pre-extracted F0
            if f0_full is not None and len(f0_full) > 0:
                # Calculate which F0 frames belong to this segment
                start_frame = int(start * self.sample_rate / fast_hop)
                end_frame = int(end * self.sample_rate / fast_hop)
                segment_f0 = f0_full[start_frame:end_frame]

                # Calculate std of voiced frames
                if len(segment_f0) > 0:
                    pitch_std = float(np.nanstd(segment_f0))
                    if np.isnan(pitch_std) or pitch_std == 0:
                        # Use mean as fallback if no variation
                        pitch_std = float(np.nanmean(segment_f0)) * 0.1 if not np.isnan(np.nanmean(segment_f0)) else 0.0
                else:
                    pitch_std = 0.0
            else:
                pitch_std = 0.0

            loudness_vals.append(loudness)
            pitch_vals.append(pitch_std)

        elapsed = time.time() - start_time
        logger.info(f"[EngagementV2] Extracted features for {len(segment_bounds)} segments in {elapsed:.1f}s")
        return loudness_vals, pitch_vals

    def calculate_engagement_scores(
        self,
        loudness_vals: List[float],
        pitch_vals: List[float]
    ) -> List[float]:
        """
        Calculate engagement scores from features.

        Args:
            loudness_vals: List of loudness values
            pitch_vals: List of pitch variation values

        Returns:
            List of engagement scores (0-1)
        """
        # Normalize features (min-max scaling)
        loud_arr = np.array(loudness_vals)
        pitch_arr = np.array(pitch_vals)

        # Avoid division by zero
        loud_range = np.ptp(loud_arr)
        pitch_range = np.ptp(pitch_arr)

        if loud_range > 1e-9:
            loud_z = (loud_arr - np.min(loud_arr)) / loud_range
        else:
            loud_z = np.zeros_like(loud_arr)

        if pitch_range > 1e-9:
            pitch_z = (pitch_arr - np.min(pitch_arr)) / pitch_range
        else:
            pitch_z = np.zeros_like(pitch_arr)

        # Weighted combination
        scores = self.loudness_weight * loud_z + self.pitch_weight * pitch_z

        return scores.tolist()

    def classify_engagement(
        self,
        score: float
    ) -> Tuple[str, float]:
        """
        Classify engagement level and confidence.

        Args:
            score: Engagement score (0-1)

        Returns:
            Tuple of (label, confidence_percentage)
        """
        if score > 0.50:
            label = "Engaging"
            confidence = score * 100
        elif score > 0.30:
            label = "Neutral"
            confidence = (1 - abs(score - 0.4) * 2) * 100
        else:
            label = "Boring"
            confidence = (1 - score) * 100

        return label, round(confidence, 1)

    def analyze_audio(
        self,
        audio_path: str,
        transcript_segments: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze engagement from audio file.

        Args:
            audio_path: Path to audio file
            transcript_segments: Optional list of transcript segments

        Returns:
            Dictionary with engagement analysis results
        """
        logger.info(f"[EngagementV2] Starting analysis for: {audio_path}")

        # Load audio
        audio, duration = self.load_audio(audio_path)

        # Create segment bounds
        if transcript_segments:
            segment_bounds = self.create_segment_bounds(transcript_segments, duration)
        else:
            segment_bounds = self._create_default_segments(duration)

        # Extract features
        loudness_vals, pitch_vals = self.extract_segment_features(audio, segment_bounds)

        # Calculate scores
        scores = self.calculate_engagement_scores(loudness_vals, pitch_vals)

        # Classify each segment
        results = []
        for i, score in enumerate(scores):
            label, confidence = self.classify_engagement(score)

            result = {
                "segment_id": i,
                "start_time": segment_bounds[i][0],
                "end_time": segment_bounds[i][1],
                "transcript": transcript_segments[i] if transcript_segments else None,
                "engagement_label": label,
                "confidence_score": confidence,
                "raw_score": round(score, 4),
                "loudness": round(loudness_vals[i], 4),
                "pitch_variation": round(pitch_vals[i], 4)
            }
            results.append(result)

        # Calculate overall engagement score
        engagement_weights = {"Engaging": 1.0, "Neutral": 0.5, "Boring": 0.0}
        overall_score = 100 * np.mean([
            engagement_weights[r["engagement_label"]] for r in results
        ])

        # Generate statistics
        labels = [r["engagement_label"] for r in results]
        stats = {
            "total_segments": len(results),
            "engaging_segments": labels.count("Engaging"),
            "neutral_segments": labels.count("Neutral"),
            "boring_segments": labels.count("Boring"),
            "average_confidence": round(np.mean([r["confidence_score"] for r in results]), 1),
            "duration_minutes": round(duration / 60, 2)
        }

        logger.info(f"[EngagementV2] Overall engagement score: {overall_score:.2f}%")

        return {
            "engagement_score": round(overall_score, 2),
            "results": results,
            "statistics": stats,
            "audio_path": audio_path,
            "duration": duration
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration info."""
        return {
            "model_type": "audio_engagement_analyzer",
            "features": ["loudness (RMS)", "pitch_variation (F0_std)"],
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "loudness_weight": self.loudness_weight,
            "pitch_weight": self.pitch_weight,
            "gpu_required": False,
            "memory_usage": "minimal (CPU only)"
        }


# Factory function
def create_engagement_analyzer() -> EngagementAnalyzerV2:
    """
    Create and configure engagement analyzer.

    Returns:
        Initialized EngagementAnalyzerV2 instance
    """
    return EngagementAnalyzerV2()


# Test function
def test_engagement_analyzer():
    """Test engagement analyzer with synthetic audio."""
    logger.info("="*80)
    logger.info("Testing Engagement Analyzer V2")
    logger.info("="*80)

    # Create analyzer
    analyzer = create_engagement_analyzer()

    # Print model info
    info = analyzer.get_model_info()
    logger.info(f"\nModel Info: {info}")

    # Create test audio (2 minutes of varying engagement)
    sr = 16000
    duration = 120  # 2 minutes
    t = np.linspace(0, duration, sr * duration)

    # Simulate speech with varying loudness and pitch
    audio = np.zeros_like(t)

    # High engagement (0-40s): loud, varied pitch
    audio[:40*sr] = 0.5 * np.sin(2 * np.pi * 200 * t[:40*sr]) + \
                     0.3 * np.sin(2 * np.pi * 350 * t[:40*sr])

    # Medium engagement (40-80s): moderate
    audio[40*sr:80*sr] = 0.3 * np.sin(2 * np.pi * 180 * t[40*sr:80*sr])

    # Low engagement (80-120s): quiet, monotone
    audio[80*sr:] = 0.1 * np.sin(2 * np.pi * 150 * t[80*sr:])

    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        sf.write(temp_path, audio, sr)

    # Test segments
    segments = [
        "This is a very engaging lecture segment with lots of energy!",
        "Here we discuss some moderate content in a neutral tone.",
        "Now we're getting into some boring monotone material..."
    ]

    # Analyze
    result = analyzer.analyze_audio(temp_path, segments)

    # Print results
    logger.info("\n" + "="*80)
    logger.info("Analysis Results:")
    logger.info("="*80)
    logger.info(f"\nOverall Engagement Score: {result['engagement_score']}%")
    logger.info(f"\nStatistics:")
    for key, value in result['statistics'].items():
        logger.info(f"  {key}: {value}")

    logger.info(f"\nSegment Details:")
    for r in result['results']:
        logger.info(f"\nSegment {r['segment_id'] + 1}: {r['engagement_label']} ({r['confidence_score']}%)")
        logger.info(f"  Loudness: {r['loudness']:.4f}, Pitch Var: {r['pitch_variation']:.4f}")

    # Cleanup
    Path(temp_path).unlink()

    logger.info("\n" + "="*80)
    logger.info("Test Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    test_engagement_analyzer()
