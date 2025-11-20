"""
Translation Module

Lightweight 4-bit GGUF model implementation for Arabic-English translation.
Optimized for code-switched Egyptian Arabic–English text.

Features:
- 4-bit quantized GGUF model for efficiency
- GPU acceleration support
- Language statistics and analysis
- Automatic translation detection
- Token limit handling for long texts

Model: ahmedheakl/arazn-llama3-english-gguf
Author: Ahmed
"""

import os
import re
import nltk
import time
import logging
from typing import Dict, Any, Optional, List
from langdetect import DetectorFactory, detect
from huggingface_hub import login, hf_hub_download

try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("[WARNING] llama-cpp-python not available. Translation functionality disabled.")
    print("[IDEA] To enable translation, install Visual Studio Build Tools and run: pip install llama-cpp-python")

# Download NLTK data silently
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except:
    pass

# Set deterministic language detection
DetectorFactory.seed = 0

# Global translator instance
_translator_instance = None

# Token limit constants - REDUCED for better reliability
MAX_TOKENS = (
    1200  # More conservative limit (was 1800) to leave more buffer for prompt overhead
)
CHUNK_OVERLAP = 30  # Reduced overlap (was 50) to fit more content per chunk

# Configure logging
logger = logging.getLogger(__name__)


class TranslationModel:
    """4-bit GGUF translation model wrapper"""

    def __init__(self, model_repo: str = "ahmedheakl/arazn-llama3-english-gguf"):
        self.model_repo = model_repo
        self.model_filename = "arazn-llama3-english-gguf-unsloth.Q4_K_M.gguf"
        self.llm = None
        self._load_model()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self):
        """Load the 4-bit GGUF translation model"""
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python not available")

        try:
            print(f"[INFO] Downloading translation model from {self.model_repo}")

            # Authenticate with Hugging Face
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                login(token=hf_token)
            else:
                print("[WARNING] HF_TOKEN not set. Public models only.")

            # Download model
            model_path = hf_hub_download(
                repo_id=self.model_repo, filename=self.model_filename
            )

            # Check GPU availability
            use_gpu = self._check_gpu_availability()
            gpu_layers = 32 if use_gpu else 0

            print(f"[TOOL] Loading model on {'GPU' if use_gpu else 'CPU'}")

            # Initialize model
            self.llm = Llama(
                model_path=model_path,
                n_threads=8,
                n_ctx=2048,
                temperature=0.1,
                verbose=False,
                n_gpu_layers=gpu_layers,
            )

            print(
                f"[OK] Translation model loaded successfully on {'GPU' if use_gpu else 'CPU'}"
            )

        except Exception as e:
            print(f"[ERROR] Failed to load translation model: {e}")
            raise e

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (improved approximation)

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # More accurate approximation based on the llama3 tokenizer
        words = len(text.split())

        # Check if text contains Arabic
        has_arabic = bool(
            re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u0870-\u089F]", text)
        )

        if has_arabic:
            # Arabic text typically uses more tokens per word - be more conservative
            estimated_tokens = int(words * 1.8)  # Increased from 1.3
        else:
            # English text uses fewer tokens per word
            estimated_tokens = int(words * 1.0)  # Increased from 0.75

        # Add a safety buffer
        estimated_tokens = int(estimated_tokens * 1.2)

        return max(estimated_tokens, 1)

    def _split_into_chunks(self, text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
        """
        Split text into chunks that fit within token limit - IMPROVED VERSION

        Args:
            text: Input text to split
            max_tokens: Maximum tokens per chunk

        Returns:
            List of text chunks
        """
        if not text.strip():
            return [text]

        # Use a more conservative token limit per chunk to ensure safety
        safe_max_tokens = max_tokens - 100  # Extra safety buffer

        # Try to split by sentences first
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback to simple splitting if NLTK fails
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed the limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if self._estimate_tokens(test_chunk) <= safe_max_tokens:
                current_chunk = test_chunk
            else:
                # Current chunk is full, save it and start new chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Check if single sentence is too long
                if self._estimate_tokens(sentence) > safe_max_tokens:
                    # Split long sentence by words with smaller chunks
                    word_chunks = self._split_by_words(
                        sentence, safe_max_tokens // 2
                    )  # Even smaller chunks
                    chunks.extend(word_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Final safety check - if chunks are still too large, split them further
        final_chunks = []
        for chunk in chunks:
            if self._estimate_tokens(chunk) > safe_max_tokens:
                # Force split by words
                word_splits = self._split_by_words(chunk, safe_max_tokens // 2)
                final_chunks.extend(word_splits)
            else:
                final_chunks.append(chunk)

        # If no chunks created (edge case), return original text truncated
        if not final_chunks:
            words = text.split()
            # Very conservative word count
            safe_word_count = safe_max_tokens // 3
            if len(words) > safe_word_count:
                final_chunks = [" ".join(words[:safe_word_count])]
            else:
                final_chunks = [text]

        return final_chunks

    def _split_by_words(self, text: str, max_tokens: int) -> List[str]:
        """
        Split text by words when sentence-level splitting isn't enough - IMPROVED

        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk

        Returns:
            List of word-level chunks
        """
        words = text.split()
        chunks = []
        current_chunk = ""

        # Use an even more conservative approach for word-level splitting
        safe_max_tokens = max_tokens - 50

        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word

            if self._estimate_tokens(test_chunk) <= safe_max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text[:500]]  # Fallback: truncate to 500 chars

    def translate(self, text: str) -> str:
        """
        Translate Arabic-English code-switched text to English

        Args:
            text: Input text to translate

        Returns:
            Translated English text
        """
        if not self.llm:
            raise RuntimeError("Translation model not loaded")

        if not text or not text.strip():
            return text

        try:
            # Check if text exceeds token limit
            estimated_tokens = self._estimate_tokens(text)

            if estimated_tokens > MAX_TOKENS:
                print(
                    f"[WARNING] Input text ({estimated_tokens} tokens) exceeds limit ({MAX_TOKENS}). Using chunked translation."
                )
                return self._translate_chunks(text)
            else:
                return self._translate_single(text)

        except Exception as e:
            if "exceed context window" in str(e) or "Requested tokens" in str(e):
                print(f"[WARNING] Context window exceeded. Attempting chunked translation.")
                try:
                    return self._translate_chunks(text)
                except Exception as chunk_e:
                    print(f"[ERROR] Chunked translation also failed: {chunk_e}")
                    return f"[Translation Error: Text too long - {chunk_e}]"
            else:
                print(f"[ERROR] Translation failed: {e}")
                return f"[Translation Error: {e}]"

    def _translate_single(self, text: str) -> str:
        """Translate a single text chunk"""
        # Use improved prompt format for better translations
        prompt = (
            "Translate the following text completely to English. Preserve all words and meaning:\n\n"
            f"Input: {text}\n"
            "English translation:"
        )

        try:
            response = self.llm(
                prompt,
                max_tokens=200,  # Increased to allow complete translations
                stop=[
                    "\n\n",
                    "Input:",
                    "English translation:",
                    "Note:",
                    "Answer:",
                ],  # Better stop tokens
                echo=False,
                temperature=0.1,  # Lower temperature for more focused output
                repeat_penalty=1.1,
            )

            result = response["choices"][0]["text"].strip()

            # Debug output
            print(f"   [SEARCH] Model response: '{result}'")

            # Clean up the result
            result = self._clean_translation_output(result)

            # If result is empty or too short, provide fallback
            if not result or len(result.strip()) < 3:
                print(f"   [WARNING] Empty/short translation, using fallback")
                result = text  # Return original text as fallback

            return result

        except Exception as e:
            print(f"   [ERROR] Translation error: {e}")
            return text  # Return original text on error

    def _translate_chunks(self, text: str) -> str:
        """
        Translate text by splitting into chunks

        Args:
            text: Input text to translate

        Returns:
            Combined translated text
        """
        chunks = self._split_into_chunks(text)
        translated_chunks = []

        print(f"[MEMO] Translating {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks, 1):
            try:
                print(f"   Chunk {i}/{len(chunks)}: {len(chunk.split())} words")
                translated_chunk = self._translate_single(chunk)
                translated_chunks.append(translated_chunk)
            except Exception as e:
                print(f"[ERROR] Failed to translate chunk {i}: {e}")
                # Keep original chunk if translation fails
                translated_chunks.append(f"[Chunk {i} translation failed: {chunk}]")

        # Combine translated chunks
        combined_translation = " ".join(translated_chunks)

        # Clean up the combined result
        combined_translation = self._clean_translation_output(combined_translation)

        return combined_translation

    def _clean_translation_output(self, text: str) -> str:
        """Clean and normalize translation output"""
        if not text:
            return text

        # Remove non-breaking spaces and other unwanted unicode characters
        text = text.replace("\xa0", " ")
        text = text.replace("\u200b", "")  # Zero-width space
        text = text.replace("\u200c", "")  # Zero-width non-joiner
        text = text.replace("\u200d", "")  # Zero-width joiner

        # Remove excessive punctuation
        text = re.sub(r"\.{3,}", ".", text)  # Replace 3+ dots with single dot
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace

        # Remove trailing punctuation if it looks excessive
        text = re.sub(r"\s*\.\s*$", "", text)  # Remove trailing dots

        # Remove common overgeneration patterns
        overgeneration_patterns = [
            r"\s*good luck.*",
            r"\s*have a nice day.*",
            r"\s*best regards.*",
            r"\s*your friend.*",
            r"\s*thank you.*",
            r"\s*\.\s*\.\s*\.",  # Multiple spaced dots
        ]

        for pattern in overgeneration_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        return text.strip()

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self.model_repo

    def is_available(self) -> bool:
        """Check if translation model is loaded and available"""
        return self.llm is not None


def calculate_language_stats(text: str) -> Dict[str, Any]:
    """
    Calculate comprehensive language statistics for text

    Args:
        text: Input text to analyze

    Returns:
        Dictionary containing language statistics
    """
    if not text or not text.strip():
        return {
            "english_percent": 100.0,
            "arabic_percent": 0.0,
            "mixed_percent": 0.0,
            "non_text_percent": 0.0,
            "sentence_count": 0,
            "total_tokens": 0,
        }

    # Tokenize text
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text.strip())

    if not tokens:
        return {
            "english_percent": 100.0,
            "arabic_percent": 0.0,
            "mixed_percent": 0.0,
            "non_text_percent": 0.0,
            "sentence_count": 0,
            "total_tokens": 0,
        }

    # Analyze token languages
    english_count = 0
    arabic_count = 0
    mixed_count = 0
    non_text_count = 0

    for token in tokens:
        if len(token) <= 1 or not any(c.isalpha() for c in token):
            non_text_count += 1
            continue

        # Arabic character detection
        has_arabic = bool(
            re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u0870-\u089F]", token)
        )
        has_english = bool(re.search(r"[a-zA-Z]", token))

        if has_arabic and has_english:
            mixed_count += 1
        elif has_arabic:
            arabic_count += 1
        elif has_english:
            english_count += 1
        else:
            non_text_count += 1

    total_tokens = len(tokens)
    sentence_count = len(re.split(r"[.!?]+", text))

    return {
        "english_percent": (
            100 * english_count / total_tokens if total_tokens > 0 else 0
        ),
        "arabic_percent": 100 * arabic_count / total_tokens if total_tokens > 0 else 0,
        "mixed_percent": 100 * mixed_count / total_tokens if total_tokens > 0 else 0,
        "non_text_percent": (
            100 * non_text_count / total_tokens if total_tokens > 0 else 0
        ),
        "sentence_count": sentence_count,
        "total_tokens": total_tokens,
    }


def load_translator() -> Optional[TranslationModel]:
    """
    Load the translation model.
    
    Returns:
        TranslationModel instance or None if unavailable
    """
    global _translator_instance
    
    if _translator_instance is not None:
        return _translator_instance
    
    if not LLAMA_CPP_AVAILABLE:
        print("[WARNING] Cannot load translator: llama-cpp-python not available")
        print("[IDEA] For full translation features, install Visual Studio Build Tools and run:")
        print("   pip install llama-cpp-python")
        return None
    
    try:
        _translator_instance = TranslationModel()
        return _translator_instance
    except Exception as e:
        print(f"[ERROR] Failed to load translation model: {e}")
        return None


def translate_text(
    text: str, model: Optional[TranslationModel] = None, handle_long_text: str = "chunk"
) -> Dict[str, Any]:
    """
    Translate Arabic-English code-switched text to English.
    
    Args:
        text: Input text to translate
        model: Translation model instance (optional)
        handle_long_text: How to handle long text ('chunk', 'truncate', 'error')
        
    Returns:
        Dictionary with translation results and metadata
    """
    
    if not text or not text.strip():
        return {
            "translated_text": "",
            "detected_language": "unknown",
            "confidence": 0.0,
            "processing_time": 0.0,
            "language_stats": {},
            "model_used": "none",
            "token_count": 0
        }
    
    # Load model if not provided
    if model is None:
        model = load_translator()
    
    # If model is not available, provide basic fallback
    if model is None or not model.is_available():
        # Basic language detection for fallback
        try:
            from langdetect import detect
            detected_lang = detect(text)
        except:
            detected_lang = "auto"
        
        # Simple Arabic script detection
        import re
        has_arabic = bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u0870-\u089F]", text))
        
        # Basic language stats
        stats = calculate_language_stats(text)
        
        return {
            "translated_text": f"[Translation unavailable - llama-cpp-python not installed] Original: {text[:200]}{'...' if len(text) > 200 else ''}",
            "detected_language": detected_lang,
            "confidence": 0.5,
            "processing_time": 0.1,
            "language_stats": stats,
            "model_used": "fallback",
            "token_count": len(text.split()),
            "error": "Translation model not available - install llama-cpp-python for full functionality"
        }
    
    # Use the actual model for translation
    start_time = time.time()
    
    try:
        # Estimate tokens
        token_count = model._estimate_tokens(text)
        
        # Handle long text based on strategy
        if token_count > MAX_TOKENS:
            logger.info(f"Text too long ({token_count} tokens), using chunking strategy")
            if handle_long_text == "error":
                raise ValueError(f"Text too long ({token_count} tokens, max {MAX_TOKENS})")
            elif handle_long_text == "truncate":
                # Truncate to fit within limits
                words = text.split()
                target_words = int(MAX_TOKENS * 0.75)  # Conservative estimate
                text = " ".join(words[:target_words])
                token_count = model._estimate_tokens(text)
                logger.info(f"Text truncated to {token_count} tokens")
                
        # Perform translation
        translated_text = model.translate(text)
        
        # Calculate language statistics
        language_stats = calculate_language_stats(text)
        
        processing_time = time.time() - start_time
        
        # Check if chunking was used
        chunks_used = 1 if token_count <= MAX_TOKENS else len(model._split_into_chunks(text))
        
        return {
            "translated_text": translated_text,
            "detected_language": language_stats.get("dominant_language", "auto"),
            "source_language": language_stats.get("dominant_language", "auto"),
            "target_language": "en",
            "confidence": 0.85,  # Default confidence for successful translation
            "processing_time": processing_time,
            "language_stats": language_stats,
            "model_used": "ahmedheakl/arazn-llama3-english-gguf",
            "token_count": token_count,
            "chunks_processed": chunks_used,
            "original_length": len(text),
            "translated_length": len(translated_text)
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Translation failed: {e}")
        
        # Fallback response
        stats = calculate_language_stats(text)
        return {
            "translated_text": f"Translation failed: {str(e)}",
            "detected_language": stats.get("dominant_language", "auto"),
            "confidence": 0.0,
            "processing_time": processing_time,
            "language_stats": stats,
            "model_used": "error_fallback",
            "token_count": len(text.split()),
            "error": str(e)
        }


# For testing and development
if __name__ == "__main__":
    print("🧪 Testing Translation Module")
    print("=" * 40)

    if not LLAMA_CPP_AVAILABLE:
        print("[ERROR] llama-cpp-python not available")
        exit(1)

    # Test text samples
    test_texts = [
        "Hello world",
        "مرحبا بالعالم",
        "Hello مرحبا this is mixed كلام",
        "The lecture today is about psychology وعلم النفس",
    ]

    # Load translator
    translator = load_translator()

    if translator and translator.is_available():
        print("[OK] Translation model loaded successfully!")

        for i, text in enumerate(test_texts, 1):
            print(f"\n🧪 Test {i}: {text}")
            result = translate_text(text, translator)
            print(f"[MEMO] Translation: {result['translated_text']}")
            print(f"[GLOBE] Primary Language: {result['detected_language']}")
            print(f"[CHART] English: {result['language_stats']['english_percent']:.1f}%")
            print(f"[CHART] Arabic: {result['language_stats']['arabic_percent']:.1f}%")
    else:
        print("[ERROR] Translation model failed to load")
        exit(1)
