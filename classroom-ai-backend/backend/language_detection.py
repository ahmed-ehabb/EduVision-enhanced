"""
Language Detection System for Classroom AI
Analyzes audio transcriptions and text to detect language usage and English proficiency
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import nltk
from langdetect import detect, detect_langs, DetectorFactory
from collections import Counter
import string

# Set seed for consistent results
DetectorFactory.seed = 0

class LanguageDetectionSystem:
    """
    Comprehensive language detection and analysis system
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # English word list (common words)
        self.english_common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 
            'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 
            'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 
            'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 
            'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 
            'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 
            'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 
            'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 
            'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 
            'us', 'is', 'was', 'are', 'been', 'has', 'had', 'were', 'said', 'each', 
            'which', 'their', 'said', 'if', 'do', 'will', 'you', 'what', 'so', 'can'
        }
        
        # Academic English words (for classroom context)
        self.academic_english_words = {
            'analyze', 'analysis', 'concept', 'theory', 'hypothesis', 'method', 'research',
            'study', 'examine', 'investigate', 'demonstrate', 'explain', 'describe', 
            'discuss', 'compare', 'contrast', 'evaluate', 'assess', 'interpret', 
            'conclude', 'summarize', 'understand', 'knowledge', 'learning', 'education',
            'student', 'teacher', 'lecture', 'lesson', 'assignment', 'homework', 
            'question', 'answer', 'problem', 'solution', 'example', 'practice',
            'important', 'significant', 'relevant', 'appropriate', 'effective',
            'accurate', 'correct', 'incorrect', 'proper', 'suitable'
        }
        
        # Language confidence thresholds
        self.confidence_thresholds = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }
        
        self.logger.info("[OK] Language Detection System initialized")
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('words', quiet=True)
        except Exception as e:
            self.logger.warning(f"Could not download NLTK data: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Remove non-alphabetic characters but keep spaces
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove extra spaces again
            text = re.sub(r'\s+', ' ', text.strip())
            
            return text
        
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            return text
    
    def detect_language_basic(self, text: str) -> Dict[str, Any]:
        """Basic language detection using langdetect"""
        try:
            if not text or len(text.strip()) < 3:
                return {
                    'detected_language': 'unknown',
                    'confidence': 0.0,
                    'is_english': False,
                    'method': 'basic_langdetect',
                    'error': 'Text too short'
                }
            
            # Clean text
            clean_text = self.preprocess_text(text)
            
            if len(clean_text) < 3:
                return {
                    'detected_language': 'unknown',
                    'confidence': 0.0,
                    'is_english': False,
                    'method': 'basic_langdetect',
                    'error': 'Text too short after cleaning'
                }
            
            # Detect language
            detected_lang = detect(clean_text)
            
            # Get detailed probabilities
            lang_probs = detect_langs(clean_text)
            english_confidence = 0.0
            
            for lang_prob in lang_probs:
                if lang_prob.lang == 'en':
                    english_confidence = lang_prob.prob
                    break
            
            return {
                'detected_language': detected_lang,
                'confidence': english_confidence if detected_lang == 'en' else 1 - english_confidence,
                'is_english': detected_lang == 'en',
                'method': 'basic_langdetect',
                'all_languages': [(lp.lang, lp.prob) for lp in lang_probs],
                'english_confidence': english_confidence
            }
        
        except Exception as e:
            self.logger.error(f"Error in basic language detection: {e}")
            return {
                'detected_language': 'error',
                'confidence': 0.0,
                'is_english': False,
                'method': 'basic_langdetect',
                'error': str(e)
            }
    
    def calculate_english_word_ratio(self, text: str) -> Dict[str, Any]:
        """Calculate the ratio of English words in the text"""
        try:
            clean_text = self.preprocess_text(text)
            words = clean_text.split()
            
            if not words:
                return {
                    'total_words': 0,
                    'english_words': 0,
                    'english_ratio': 0.0,
                    'common_word_ratio': 0.0,
                    'academic_word_ratio': 0.0
                }
            
            english_count = 0
            common_count = 0
            academic_count = 0
            
            for word in words:
                if word in self.english_common_words:
                    english_count += 1
                    common_count += 1
                elif word in self.academic_english_words:
                    english_count += 1
                    academic_count += 1
                elif self._is_english_word(word):
                    english_count += 1
            
            total_words = len(words)
            english_ratio = english_count / total_words if total_words > 0 else 0
            common_ratio = common_count / total_words if total_words > 0 else 0
            academic_ratio = academic_count / total_words if total_words > 0 else 0
            
            return {
                'total_words': total_words,
                'english_words': english_count,
                'english_ratio': english_ratio,
                'common_word_ratio': common_ratio,
                'academic_word_ratio': academic_ratio,
                'is_primarily_english': english_ratio > 0.7
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating English word ratio: {e}")
            return {
                'total_words': 0,
                'english_words': 0,
                'english_ratio': 0.0,
                'error': str(e)
            }
    
    def _is_english_word(self, word: str) -> bool:
        """Check if a word is likely English using various heuristics"""
        try:
            # Simple heuristics for English words
            if len(word) < 2:
                return False
            
            # Check for common English patterns
            english_patterns = [
                r'.*ing$',  # -ing endings
                r'.*ed$',   # -ed endings
                r'.*ly$',   # -ly endings
                r'.*tion$', # -tion endings
                r'.*able$', # -able endings
                r'.*ful$',  # -ful endings
            ]
            
            for pattern in english_patterns:
                if re.match(pattern, word):
                    return True
            
            # Check character frequency (English has specific patterns)
            vowels = sum(1 for c in word if c in 'aeiou')
            consonants = len(word) - vowels
            
            # English typically has a good vowel/consonant balance
            if len(word) > 3 and vowels > 0 and consonants > 0:
                vowel_ratio = vowels / len(word)
                if 0.2 <= vowel_ratio <= 0.6:  # Typical English range
                    return True
            
            return False
        
        except Exception:
            return False
    
    def analyze_classroom_language_usage(self, text: str, context: str = "lecture") -> Dict[str, Any]:
        """Comprehensive analysis of language usage in classroom context"""
        try:
            # Basic language detection
            lang_detection = self.detect_language_basic(text)
            
            # English word ratio analysis
            word_analysis = self.calculate_english_word_ratio(text)
            
            # Classroom-specific analysis
            classroom_analysis = self._analyze_classroom_vocabulary(text, context)
            
            # Calculate overall English proficiency score
            proficiency_score = self._calculate_proficiency_score(
                lang_detection, word_analysis, classroom_analysis
            )
            
            # Determine language compliance
            is_compliant = self._determine_language_compliance(
                lang_detection, word_analysis, proficiency_score
            )
            
            return {
                'timestamp': datetime.now(),
                'text_length': len(text),
                'context': context,
                'language_detection': lang_detection,
                'word_analysis': word_analysis,
                'classroom_analysis': classroom_analysis,
                'proficiency_score': proficiency_score,
                'is_english_compliant': is_compliant,
                'recommendations': self._generate_recommendations(
                    lang_detection, word_analysis, proficiency_score
                )
            }
        
        except Exception as e:
            self.logger.error(f"Error in classroom language analysis: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e),
                'is_english_compliant': False
            }
    
    def _analyze_classroom_vocabulary(self, text: str, context: str) -> Dict[str, Any]:
        """Analyze vocabulary specific to classroom context"""
        try:
            clean_text = self.preprocess_text(text)
            words = clean_text.split()
            
            # Count academic vocabulary
            academic_words = [w for w in words if w in self.academic_english_words]
            
            # Educational keywords
            educational_keywords = {
                'lecture': ['explain', 'understand', 'learn', 'study', 'example', 'concept'],
                'discussion': ['discuss', 'opinion', 'think', 'believe', 'agree', 'disagree'],
                'assignment': ['homework', 'assignment', 'task', 'complete', 'submit'],
                'question': ['question', 'answer', 'help', 'clarify', 'explain']
            }
            
            context_words = educational_keywords.get(context, [])
            context_word_count = sum(1 for w in words if w in context_words)
            
            # Calculate complexity metrics
            avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
            long_words = sum(1 for w in words if len(w) > 6)
            
            return {
                'academic_word_count': len(academic_words),
                'context_word_count': context_word_count,
                'avg_word_length': avg_word_length,
                'long_word_count': long_words,
                'vocabulary_diversity': len(set(words)) / len(words) if words else 0,
                'context_relevance': context_word_count / len(words) if words else 0
            }
        
        except Exception as e:
            self.logger.error(f"Error analyzing classroom vocabulary: {e}")
            return {}
    
    def _calculate_proficiency_score(self, lang_detection: Dict, word_analysis: Dict, classroom_analysis: Dict) -> float:
        """Calculate overall English proficiency score (0-100)"""
        try:
            score = 0.0
            
            # Language detection score (40% weight)
            if lang_detection.get('is_english', False):
                lang_score = lang_detection.get('english_confidence', 0) * 100
                score += lang_score * 0.4
            
            # Word analysis score (40% weight)
            english_ratio = word_analysis.get('english_ratio', 0)
            word_score = english_ratio * 100
            score += word_score * 0.4
            
            # Classroom vocabulary score (20% weight)
            academic_boost = min(classroom_analysis.get('academic_word_count', 0) * 2, 20)
            context_boost = classroom_analysis.get('context_relevance', 0) * 100 * 0.1
            classroom_score = academic_boost + context_boost
            score += classroom_score * 0.2
            
            return min(100.0, max(0.0, score))
        
        except Exception as e:
            self.logger.error(f"Error calculating proficiency score: {e}")
            return 0.0
    
    def _determine_language_compliance(self, lang_detection: Dict, word_analysis: Dict, proficiency_score: float) -> bool:
        """Determine if the language usage is compliant with English requirements"""
        try:
            # Multiple criteria for compliance
            criteria = [
                lang_detection.get('is_english', False),
                word_analysis.get('english_ratio', 0) > 0.7,
                proficiency_score > 70,
                lang_detection.get('english_confidence', 0) > 0.6
            ]
            
            # Must meet at least 3 out of 4 criteria
            return sum(criteria) >= 3
        
        except Exception as e:
            self.logger.error(f"Error determining compliance: {e}")
            return False
    
    def _generate_recommendations(self, lang_detection: Dict, word_analysis: Dict, proficiency_score: float) -> List[str]:
        """Generate recommendations for improving English usage"""
        try:
            recommendations = []
            
            if not lang_detection.get('is_english', False):
                recommendations.append("Consider using English as the primary language of instruction")
            
            if word_analysis.get('english_ratio', 0) < 0.7:
                recommendations.append("Increase the use of English vocabulary")
            
            if proficiency_score < 70:
                recommendations.append("Work on improving English proficiency for clearer communication")
            
            if word_analysis.get('academic_word_ratio', 0) < 0.1:
                recommendations.append("Incorporate more academic vocabulary for educational context")
            
            if not recommendations:
                recommendations.append("Excellent English usage! Continue with current approach")
            
            return recommendations
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations"]
    
    def analyze_language_trends(self, analysis_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze language usage trends over time"""
        try:
            if not analysis_history:
                return {"error": "No analysis history provided"}
            
            # Extract metrics over time
            proficiency_scores = [a.get('proficiency_score', 0) for a in analysis_history]
            english_ratios = [a.get('word_analysis', {}).get('english_ratio', 0) for a in analysis_history]
            compliance_rate = sum(1 for a in analysis_history if a.get('is_english_compliant', False)) / len(analysis_history)
            
            # Calculate trends
            if len(proficiency_scores) > 1:
                proficiency_trend = proficiency_scores[-1] - proficiency_scores[0]
                english_ratio_trend = english_ratios[-1] - english_ratios[0]
            else:
                proficiency_trend = 0
                english_ratio_trend = 0
            
            return {
                'total_analyses': len(analysis_history),
                'average_proficiency': sum(proficiency_scores) / len(proficiency_scores),
                'average_english_ratio': sum(english_ratios) / len(english_ratios),
                'compliance_rate': compliance_rate,
                'proficiency_trend': proficiency_trend,
                'english_ratio_trend': english_ratio_trend,
                'trend_analysis': {
                    'improving': proficiency_trend > 5,
                    'stable': -5 <= proficiency_trend <= 5,
                    'declining': proficiency_trend < -5
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return {"error": str(e)} 