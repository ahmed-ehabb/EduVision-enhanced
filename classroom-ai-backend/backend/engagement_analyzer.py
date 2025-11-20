"""
Engagement Analysis Module for Classroom AI System.

This module analyzes audio and video features to determine engagement levels
in real-time and recorded sessions.

Features:
- Real-time audio analysis (loudness, pitch, speaking rate)
- Face detection and tracking
- Gaze direction estimation
- Emotion recognition
- Weighted engagement scoring
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import tempfile
import subprocess
from pathlib import Path
import asyncio
from datetime import datetime
import torch
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
from .gpu_config import gpu_config
from .error_handler import handle_model_error

# Audio processing libraries
try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    print(
        "Warning: Audio processing libraries not available. "
        "Install with: pip install librosa soundfile"
    )
    AUDIO_PROCESSING_AVAILABLE = False

# Video processing libraries
try:
    import cv2
    import mediapipe as mp
    VIDEO_PROCESSING_AVAILABLE = True
except ImportError:
    print(
        "Warning: Video processing libraries not available. "
        "Install with: pip install opencv-python mediapipe"
    )
    VIDEO_PROCESSING_AVAILABLE = False

from .face_recognition_model import FaceRecognitionModel
from .engagement_classifier import EngagementClassifier
from .utils.logger import get_logger

logger = get_logger(__name__)


class EngagementAnalyzer:
    """
    Analyzes engagement levels using audio and video features in real-time
    and for recorded sessions.
    """

    def __init__(self, model_path: str = 'models/engagement'):
        self.model_path = Path(model_path)
        self.feature_extractor = None
        self.model = None
        self.device = gpu_config.device
        self.config = gpu_config.get_model_config('engagement')
        
    async def initialize(self):
        """Initialize engagement analysis models"""
        try:
            with gpu_config.device_context('engagement'):
                # Load DAISEE engagement model
                model_name = "Shadhujan/daisee_engagement_model_050725l.h5"
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Move model to GPU if available
                self.model = self.model.to(self.device)
                
                # Enable FP16 if supported
                if self.config['use_fp16'] and self.device.type == 'cuda':
                    self.model = self.model.half()
                    
                # Enable evaluation mode
                self.model.eval()
                
        except Exception as e:
            handle_model_error("Engagement Analyzer", str(e))
            
    async def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze a video frame for engagement level"""
        if self.model is None:
            await self.initialize()
            
        try:
            # Prepare input
            inputs = self.feature_extractor(
                frame,
                return_tensors="pt"
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                if self.config['use_fp16']:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
                    
            # Process outputs
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            engagement_score = probabilities[0].cpu().numpy()
            
            # Map scores to engagement levels
            engagement_levels = ['disengaged', 'low', 'medium', 'high']
            level_idx = np.argmax(engagement_score)
            
            return {
                'score': float(engagement_score[level_idx]),
                'level': engagement_levels[level_idx],
                'scores': {
                    level: float(score)
                    for level, score in zip(engagement_levels, engagement_score)
                }
            }
            
        except Exception as e:
            handle_model_error("Frame Analysis", str(e))
            return {
                'score': 0.0,
                'level': 'unknown',
                'scores': {}
            }
            
    def cleanup(self):
        """Release GPU resources"""
        if self.model is not None:
            self.model.cpu()
        gpu_config.cleanup()

    async def analyze_realtime(
        self,
        audio_chunk: Optional[np.ndarray] = None,
        video_frame: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze engagement from real-time audio and video input.
        
        Args:
            audio_chunk: New audio data chunk (optional)
            video_frame: New video frame (optional)
            
        Returns:
            Dict with current engagement metrics
        """
        current_time = datetime.now()
        audio_features = {}
        video_features = {}
        
        # Process audio if available
        if audio_chunk is not None and AUDIO_PROCESSING_AVAILABLE:
            # Extract audio features
            audio_features = await self._extract_realtime_audio_features(audio_chunk)
        
        # Process video if available
        if video_frame is not None and VIDEO_PROCESSING_AVAILABLE:
            # Extract video features
            video_features = await self.analyze_frame(video_frame)
        
        # Calculate engagement score
        engagement_score = self._calculate_realtime_engagement(
            audio_features,
            video_features
        )
        
        # Generate detailed metrics
        metrics = {
            "timestamp": current_time.isoformat(),
            "engagement_score": engagement_score,
            "audio_metrics": audio_features,
            "video_metrics": video_features,
            "attention_level": self._classify_attention(engagement_score)
        }
        
        return metrics

    async def _extract_realtime_audio_features(
        self,
        audio_chunk: np.ndarray
    ) -> Dict[str, float]:
        """Extract features from the current audio chunk."""
        try:
            # Calculate RMS energy (loudness)
            rms = np.sqrt(np.mean(np.square(audio_chunk)))
            
            # Calculate pitch features
            if len(audio_chunk) >= 2048:  # Minimum size for pitch detection
                pitches, magnitudes = librosa.piptrack(
                    y=audio_chunk,
                    sr=16000,
                    hop_length=512
                )
                pitch_mean = np.mean(pitches[magnitudes > np.max(magnitudes)/2])
                pitch_std = np.std(pitches[magnitudes > np.max(magnitudes)/2])
            else:
                pitch_mean = 0.0
                pitch_std = 0.0
            
            # Estimate speaking rate
            if len(audio_chunk) >= 4096:
                onset_env = librosa.onset.onset_strength(
                    y=audio_chunk,
                    sr=16000
                )
                speaking_rate = len(librosa.onset.onset_detect(
                    onset_envelope=onset_env,
                    sr=16000
                )) / (len(audio_chunk) / 16000)
            else:
                speaking_rate = 0.0
            
            return {
                "loudness": float(rms),
                "pitch_mean": float(pitch_mean),
                "pitch_variation": float(pitch_std),
                "speaking_rate": float(speaking_rate)
            }
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {
                "loudness": 0.0,
                "pitch_mean": 0.0,
                "pitch_variation": 0.0,
                "speaking_rate": 0.0
            }

    def _calculate_realtime_engagement(
        self,
        audio_features: Dict[str, float],
        video_features: Dict[str, float]
    ) -> float:
        """Calculate overall engagement score from current features."""
        score = 0.0
        
        # Audio component
        if audio_features:
            audio_score = (
                audio_features["loudness"] * 0.3 +
                audio_features["pitch_variation"] * 0.2 +
                audio_features["speaking_rate"] * 0.1
            )
            score += audio_score
        
        # Video component
        if video_features:
            video_score = (
                video_features["score"] * 0.1
            )
            score += video_score
        
        return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1

    def _classify_attention(self, score: float) -> str:
        """Classify attention level based on engagement score."""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"

    async def analyze_recording(
        self,
        audio_path: str,
        transcript_segments: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze engagement levels in recorded audio.
        
        Args:
            audio_path: Path to audio file
            transcript_segments: Optional transcript segments
            
        Returns:
            Dict with engagement analysis results
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            return {
                "error": "Audio processing libraries not available",
                "engagement_results": [],
                "engagement_score": 0,
                "engagement_stats": {}
            }

        try:
            logger.info(f"Starting engagement analysis for: {audio_path}")
            
            # Load and preprocess audio
            audio_data, duration = await self._load_audio(audio_path)
            logger.info(f"Audio loaded: {duration:.2f}s duration")
            
            if len(audio_data) == 0:
                return {
                    "error": "Audio file is empty or corrupted",
                    "engagement_results": [],
                    "engagement_score": 0,
                    "engagement_stats": {}
                }
            
            # Get segment boundaries
            if transcript_segments:
                segment_bounds = self._estimate_segment_bounds(
                    transcript_segments,
                    duration
                )
            else:
                segment_bounds = self._create_default_segments(duration)
            
            # Process each segment
            engagement_results = []
            for i, (start, end) in enumerate(segment_bounds):
                segment_audio = audio_data[
                    int(start * 16000):int(end * 16000)
                ]
                
                # Get features for segment
                features = await self._extract_realtime_audio_features(segment_audio)
                
                # Calculate engagement score
                score = self._calculate_realtime_engagement(features, {})
                
                engagement_results.append({
                    "segment_id": i,
                    "start_time": start,
                    "end_time": end,
                    "engagement_score": score,
                    "attention_level": self._classify_attention(score),
                    "features": features,
                    "transcript": transcript_segments[i] if transcript_segments else None
                })
            
            # Calculate overall stats
            engagement_stats = self._generate_engagement_stats(engagement_results)
            final_score = self._calculate_final_score(engagement_results)
            
            return {
                "engagement_results": engagement_results,
                "engagement_score": final_score,
                "engagement_stats": engagement_stats
            }
            
        except Exception as e:
            logger.error(f"Engagement analysis failed: {e}")
            return {
                "error": str(e),
                "engagement_results": [],
                "engagement_score": 0,
                "engagement_stats": {}
            }

    async def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """Load and preprocess audio file."""
        try:
            # Load audio
            audio_data, sr = librosa.load(
                audio_path,
                sr=16000,
                mono=True
            )
            
            duration = len(audio_data) / sr
            return audio_data, duration
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return np.array([]), 0.0

    def _estimate_segment_bounds(
        self,
        transcript_segments: List[str],
        total_duration: float
    ) -> List[Tuple[float, float]]:
        """Estimate time bounds for transcript segments."""
        if not transcript_segments:
            return self._create_default_segments(total_duration)
        
        # Estimate segment durations based on word count
        word_counts = [len(seg.split()) for seg in transcript_segments]
        total_words = sum(word_counts)
        
        bounds = []
        current_time = 0.0
        
        for word_count in word_counts:
            segment_duration = (word_count / total_words) * total_duration
            bounds.append((current_time, current_time + segment_duration))
            current_time += segment_duration
        
        return bounds

    def _create_default_segments(
        self,
        duration: float,
        segment_length: float = 30.0
    ) -> List[Tuple[float, float]]:
        """Create default fixed-length segments."""
        segments = []
        current_time = 0.0
        
        while current_time < duration:
            end_time = min(current_time + segment_length, duration)
            segments.append((current_time, end_time))
            current_time = end_time
        
        return segments

    def _generate_engagement_stats(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary statistics from engagement results."""
        if not results:
            return {}
        
        scores = [r["engagement_score"] for r in results]
        attention_levels = [r["attention_level"] for r in results]
        
        return {
            "mean_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "attention_distribution": {
                "high": attention_levels.count("high") / len(attention_levels),
                "medium": attention_levels.count("medium") / len(attention_levels),
                "low": attention_levels.count("low") / len(attention_levels)
            }
        }

    def _calculate_final_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate final engagement score from all results."""
        if not results:
            return 0.0
        
        # Weight recent segments more heavily
        weights = np.linspace(0.5, 1.0, len(results))
        scores = np.array([r["engagement_score"] for r in results])
        
        return float(np.average(scores, weights=weights))

    def generate_engagement_feedback(
        self,
        engagement_score: float,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """Generate feedback based on engagement score."""
        level = self._classify_attention(engagement_score)
        
        feedback = {
            "level": level,
            "score": engagement_score,
            "summary": f"Engagement level is {level} ({engagement_score:.2f})"
        }
        
        if detailed:
            if level == "high":
                feedback["suggestions"] = [
                    "Maintain current engagement strategies",
                    "Consider introducing more challenging content",
                    "Use this high engagement for complex topics"
                ]
            elif level == "medium":
                feedback["suggestions"] = [
                    "Try increasing interactive elements",
                    "Check for understanding more frequently",
                    "Consider varying teaching methods"
                ]
            else:
                feedback["suggestions"] = [
                    "Take a short break",
                    "Switch to more interactive content",
                    "Ask questions to increase participation",
                    "Consider simplifying current content"
                ]
        
        return feedback


def get_engagement_analyzer() -> EngagementAnalyzer:
    """Get a configured instance of the engagement analyzer."""
    return EngagementAnalyzer()
