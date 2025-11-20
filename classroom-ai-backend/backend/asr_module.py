"""
Automatic Speech Recognition (ASR) Module

This module provides speech-to-text functionality using the Whisper model
specifically fine-tuned for Arabic-English mixed content.
"""

import time
from typing import Optional, Dict, Any
import os
import tempfile
import numpy as np
import traceback
try:
    import torch
    import soundfile as sf
    from transformers import (
        pipeline,
        AutoModelForSpeechSeq2Seq,
        WhisperFeatureExtractor,
        WhisperTokenizer,
        WhisperProcessor
    )
    # Try to import librosa as fallback for resampling
    try:
        import librosa
        LIBROSA_AVAILABLE = True
    except ImportError:
        LIBROSA_AVAILABLE = False
        print("[WARNING] librosa not available - ffmpeg will be used for resampling")
    
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"[WARNING] ASR dependencies not available: {e}")

try:
    from deepmultilingualpunctuation import PunctuationModel
    PUNCTUATION_AVAILABLE = True
except ImportError:
    PUNCTUATION_AVAILABLE = False
    print(
        "[WARNING] Punctuation restoration not available. "
        "Install with: pip install deepmultilingualpunctuation"
    )


class ASRModule:
    """ASR model wrapper with on-demand punctuation restoration."""

    def __init__(
        self,
        model_id: str = "ahmedheakl/arazn-whisper-small-v2",
        enable_punctuation: bool = True,
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ):
        """Initializes the ASR model.

        Args:
            model_id: HuggingFace model ID
            enable_punctuation: Whether to enable punctuation restoration
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            compute_type: Compute precision ('float32', 'float16', 'int8', or None for default)
        """
        self.model_id = model_id
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type
        self.pipeline = None
        self.punctuation_model = None
        self.enable_punctuation = enable_punctuation and PUNCTUATION_AVAILABLE
        self._load_model()

    def _load_model(self):
        """Load the ASR model for direct generation (not pipeline)."""
        if not DEPENDENCIES_AVAILABLE:
            return
        try:
            compute_info = f" ({self.compute_type})" if self.compute_type else ""
            print(f"[TOOL] Loading ASR model: {self.model_id} on {self.device}{compute_info}")

            # Determine dtype
            dtype = torch.float32  # default
            if self.compute_type == "float16":
                dtype = torch.float16
            elif self.compute_type == "int8":
                dtype = torch.float16  # Use FP16 as base for INT8

            # Load model with proper BitsAndBytesConfig for INT8
            if self.compute_type == "int8":
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )

                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id,
                    dtype=dtype,  # Use 'dtype' instead of deprecated 'torch_dtype'
                    low_cpu_mem_usage=True
                ).to(self.device)

            # Load processor
            self.processor = WhisperProcessor.from_pretrained(self.model_id)

            # Store for direct use (no pipeline)
            self.pipeline = None  # We'll use model.generate() directly

            print("[OK] ASR model loaded successfully (using native generate method)")
        except Exception as e:
            print(f"[ERROR] Failed to load ASR model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.processor = None

    def _ensure_punctuation_model_loaded(self) -> bool:
        """Load the punctuation model if it hasn't been loaded yet."""
        if not self.enable_punctuation or self.punctuation_model:
            return self.enable_punctuation
        try:
            print("[TOOL] Loading punctuation restoration model...")
            self.punctuation_model = PunctuationModel(
                model="oliverguhr/fullstop-punctuation-multilang-large"
            )
            print("[OK] Punctuation model loaded successfully.")
            return True
        except Exception as e:
            print(f"[WARNING] Failed to load punctuation model: {e}")
            self.enable_punctuation = False
            return False

    def _restore_punctuation(self, text: str) -> str:
        """Restore punctuation in the given text."""
        if not self._ensure_punctuation_model_loaded():
            return text
        try:
            return self.punctuation_model.restore_punctuation(text)
        except Exception as e:
            print(f"[WARNING] Punctuation restoration failed: {e}")
            return text

    def _resample_audio_fallback(self, audio_data, sr: int, target_sr: int = 16000):
        """
        Fallback resampling using librosa when ffmpeg is not available.
        Returns resampled audio data.
        """
        if not LIBROSA_AVAILABLE:
            print("[ERROR] librosa not available for fallback resampling")
            return None, sr
            
        try:
            print(f"[SEARCH] Using librosa fallback for resampling {sr}Hz -> {target_sr}Hz")
            
            # Resample if needed
            if sr != target_sr:
                resampled_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            else:
                resampled_data = audio_data
            
            # Convert to mono if stereo
            if len(resampled_data.shape) > 1:
                print(f"[SEARCH] Converting to mono using librosa")
                resampled_data = librosa.to_mono(resampled_data.T)  # librosa expects (channels, samples)
            
            return resampled_data, target_sr
            
        except Exception as e:
            print(f"[ERROR] librosa fallback resampling failed: {e}")
            return None, sr

    def _resample_audio_with_ffmpeg(self, input_path: str, target_sr: int = 16000) -> str:
        """
        Resample audio to target sample rate and convert to mono using ffmpeg.
        Returns path to the resampled file.
        """
        import subprocess
        import tempfile
        import os
        
        # Create temporary file for resampled audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            # Use ffmpeg to resample and convert to mono
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-i', input_path,  # input file
                '-ar', str(target_sr),  # set sample rate
                '-ac', '1',  # convert to mono (1 channel)
                '-acodec', 'pcm_s16le',  # use PCM 16-bit encoding
                output_path
            ]
            
            print(f"[SEARCH] Running ffmpeg: {' '.join(cmd)}")
            
            # Run ffmpeg command
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60  # 60 second timeout
            )
            
            if result.returncode != 0:
                print(f"[ERROR] ffmpeg failed with return code {result.returncode}")
                print(f"[ERROR] ffmpeg stderr: {result.stderr}")
                # Clean up failed output file
                try:
                    os.unlink(output_path)
                except:
                    pass
                return None
            
            print(f"[OK] ffmpeg resampling successful: {output_path}")
            return output_path
            
        except subprocess.TimeoutExpired:
            print(f"[ERROR] ffmpeg timeout after 60 seconds")
            try:
                os.unlink(output_path)
            except:
                pass
            return None
        except FileNotFoundError:
            print(f"[ERROR] ffmpeg not found. Please install ffmpeg and add it to PATH")
            try:
                os.unlink(output_path)
            except:
                pass
            return None
        except Exception as e:
            print(f"[ERROR] ffmpeg error: {e}")
            try:
                os.unlink(output_path)
            except:
                pass
            return None

    def transcribe(
        self, audio_path: str, add_punctuation: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio file to text with optional punctuation restoration.
        Optimized: No WAV conversion, in-memory preprocessing, punctuation at end.
        """
        if not self.is_available():
            return None

        print(f"[SEARCH] ASR Debug: Processing {audio_path}")
        print(f"[SEARCH] File exists: {os.path.exists(audio_path)}")
        print(f"[SEARCH] File size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'} bytes")

        try:
            import librosa
            import numpy as np

            # Load audio directly (supports MP3, WAV, etc.)
            # Force mono and resample in one step - much faster!
            print(f"[SEARCH] Loading audio with librosa (direct mono conversion & resampling)...")
            audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(audio_data) / sr
            print(f"[SEARCH] Loaded: {duration:.2f}s ({duration/60:.2f}min), 16000Hz, mono")

            # Use chunked transcription for long audio files
            if duration > 120:  # 2 minutes
                print(f"[SEARCH] Using chunked transcription for long audio ({duration:.2f}s)")
                return self._transcribe_chunked_optimized(audio_data, sr, add_punctuation)
            else:
                print(f"[SEARCH] Using single transcription for short audio ({duration:.2f}s)")
                return self._transcribe_single_optimized(audio_data, sr, add_punctuation)

        except Exception as e:
            print(f"[ERROR] Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _transcribe_single(self, audio_path: str, add_punctuation: bool = True) -> Optional[Dict[str, Any]]:
        """Transcribe a single audio file without chunking."""
        try:
            start_time = time.time()
            print(f"[SEARCH] Calling Whisper pipeline on {audio_path}")
            result = self.pipeline(audio_path)
            print(f"[SEARCH] Pipeline result: {result}")
            processing_time = time.time() - start_time
            transcript = result.get("text", "").strip()

            output = {
                "text": transcript,
                "punctuated_text": transcript,
                "processing_time": round(processing_time, 2),
                "punctuation_time": 0,
                "model": self.model_id,
                "chunks_processed": 1
            }

            if add_punctuation and transcript:
                punc_start = time.time()
                output["punctuated_text"] = self._restore_punctuation(transcript)
                output["punctuation_time"] = round(time.time() - punc_start, 2)

            return output
        except Exception as e:
            print(f"[ERROR] Single transcription failed: {e}")
            return None

    def _transcribe_chunked(self, audio_path: str, audio_data, sr: int, add_punctuation: bool = True) -> Optional[Dict[str, Any]]:
        """Transcribe long audio by chunking it into smaller pieces."""
        try:
            import tempfile
            import numpy as np
            
            start_time = time.time()
            duration = len(audio_data) / sr
            chunk_duration = 120  # 120 seconds (2 min) per chunk for faster processing
            num_chunks = int(np.ceil(duration / chunk_duration))
            
            print(f"[SEARCH] Chunking audio: {duration:.2f}s into {num_chunks} chunks of {chunk_duration}s each")
            
            full_text = ""
            full_raw_text = ""
            total_processing_time = 0
            
            for i in range(num_chunks):
                start_sample = i * chunk_duration * sr
                end_sample = min((i + 1) * chunk_duration * sr, len(audio_data))
                chunk_data = audio_data[start_sample:end_sample]
                
                # Save chunk to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as chunk_file:
                    chunk_filename = chunk_file.name
                    sf.write(chunk_filename, chunk_data, sr)
                
                try:
                    # Transcribe chunk
                    chunk_result = self.pipeline(chunk_filename)
                finally:
                    # Clean up temp file - ensure it's closed first
                    try:
                        os.unlink(chunk_filename)
                    except (OSError, PermissionError) as e:
                        print(f"[WARNING] Warning: Could not delete temp file {chunk_filename}: {e}")
                        # Try again after a brief delay
                        time.sleep(0.1)
                        try:
                            os.unlink(chunk_filename)
                        except:
                            pass  # Give up if still can't delete
                
                if chunk_result:
                    chunk_text = chunk_result.get("text", "").strip()
                    if chunk_text:
                        full_text += chunk_text + " "
                        full_raw_text += chunk_text + " "
                        print(f"[SEARCH] Chunk {i+1}/{num_chunks}: {len(chunk_text)} chars")
                
                total_processing_time += 1.0  # Estimate processing time per chunk
            
            # Clean up text
            full_text = full_text.strip()
            full_raw_text = full_raw_text.strip()
            
            output = {
                "text": full_raw_text,
                "punctuated_text": full_text,
                "processing_time": round(time.time() - start_time, 2),
                "punctuation_time": 0,
                "model": self.model_id,
                "chunks_processed": num_chunks,
                "duration": duration
            }

            # Add punctuation to the combined text
            if add_punctuation and full_text:
                punc_start = time.time()
                output["punctuated_text"] = self._restore_punctuation(full_text)
                output["punctuation_time"] = round(time.time() - punc_start, 2)

            print(f"[SEARCH] Chunked transcription completed: {len(full_text)} chars from {num_chunks} chunks")
            return output
            
        except Exception as e:
            import traceback
            print(f"[ERROR] Chunked transcription failed: {e}")
            print(f"[SEARCH] Error details: {traceback.format_exc()}")
            return None

    def cleanup(self):
        """Release GPU memory by deleting model references."""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
                self.processor = None
            if hasattr(self, 'punctuation_model') and self.punctuation_model is not None:
                del self.punctuation_model
                self.punctuation_model = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[CLEANUP] ASR model resources released")
        except Exception as e:
            print(f"[WARNING] ASR cleanup error: {e}")

    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self.model_id

    def is_available(self) -> bool:
        """Check if ASR model is loaded and available."""
        return self.model is not None and self.processor is not None

    def is_punctuation_available(self) -> bool:
        """Check if punctuation restoration is configured and available."""
        return self.enable_punctuation

    def _transcribe_single_optimized(self, audio_data, sr: int, add_punctuation: bool = True) -> Optional[Dict[str, Any]]:
        """Transcribe audio array directly using model.generate() - Whisper's native method."""
        try:
            import numpy as np
            start_time = time.time()

            # Process audio with Whisper processor
            inputs = self.processor(audio_data, sampling_rate=sr, return_tensors="pt")

            # Convert to same dtype as model (important for quantized models!)
            input_features = inputs.input_features
            if self.model.dtype == torch.float16:
                input_features = input_features.half()
            input_features = input_features.to(self.model.device)

            # Generate transcription using Whisper's native generate method with optimizations
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*attention mask.*")
                predicted_ids = self.model.generate(
                    input_features,
                    language="en",  # Force English for faster inference
                    task="transcribe",  # Transcription task
                    num_beams=1,  # Greedy decoding (faster than beam search)
                    do_sample=False,  # Deterministic output
                    use_cache=True  # Enable KV cache for faster generation
                )
                transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            processing_time = time.time() - start_time

            # Punctuate ONCE at the end (not per chunk)
            punctuated_text = transcript
            punctuation_time = 0
            if add_punctuation and transcript:
                punc_start = time.time()
                punctuated_text = self._restore_punctuation(transcript)
                punctuation_time = time.time() - punc_start

            return {
                "text": transcript,
                "punctuated_text": punctuated_text,
                "processing_time": round(processing_time, 2),
                "punctuation_time": round(punctuation_time, 2),
                "model": self.model_id,
                "chunks_processed": 1
            }

        except Exception as e:
            print(f"[ERROR] Single transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _transcribe_chunked_optimized(self, audio_data, sr: int, add_punctuation: bool = True) -> Optional[Dict[str, Any]]:
        """Transcribe long audio using Whisper's native generate() method with chunking."""
        try:
            import numpy as np

            start_time = time.time()
            duration = len(audio_data) / sr
            chunk_duration = 120  # 2 minutes per chunk
            num_chunks = int(np.ceil(duration / chunk_duration))

            print(f"[SEARCH] Chunking audio in-memory: {duration:.2f}s into {num_chunks} chunks of {chunk_duration}s each")
            print(f"[SEARCH] Using Whisper's native generate() method (no pipeline warnings)")

            full_text = ""

            for i in range(num_chunks):
                start_sample = int(i * chunk_duration * sr)
                end_sample = int(min((i + 1) * chunk_duration * sr, len(audio_data)))
                chunk_data = audio_data[start_sample:end_sample]

                # Process chunk with Whisper processor
                inputs = self.processor(chunk_data, sampling_rate=sr, return_tensors="pt")

                # Convert to same dtype as model (important for quantized models!)
                input_features = inputs.input_features
                if self.model.dtype == torch.float16:
                    input_features = input_features.half()
                input_features = input_features.to(self.model.device)

                # Use Whisper's native generate method with optimizations
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*attention mask.*")
                    predicted_ids = self.model.generate(
                        input_features,
                        language="en",  # Force English for faster inference
                        task="transcribe",  # Transcription task
                        num_beams=1,  # Greedy decoding (faster than beam search)
                        do_sample=False,  # Deterministic output
                        use_cache=True  # Enable KV cache for faster generation
                    )
                    chunk_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

                if chunk_text:
                    full_text += chunk_text + " "
                    print(f"[SEARCH] Chunk {i+1}/{num_chunks}: {len(chunk_text)} chars")

            # Clean up text
            full_text = full_text.strip()
            processing_time = time.time() - start_time

            # Punctuate ONCE at the end for entire transcript (much faster!)
            punctuated_text = full_text
            punctuation_time = 0
            if add_punctuation and full_text:
                print(f"[SEARCH] Applying punctuation to complete transcript ({len(full_text)} chars)...")
                punc_start = time.time()
                punctuated_text = self._restore_punctuation(full_text)
                punctuation_time = time.time() - punc_start
                print(f"[SEARCH] Punctuation applied in {punctuation_time:.2f}s")

            return {
                "text": full_text,
                "punctuated_text": punctuated_text,
                "processing_time": round(processing_time, 2),
                "punctuation_time": round(punctuation_time, 2),
                "model": self.model_id,
                "chunks_processed": num_chunks
            }

        except Exception as e:
            print(f"[ERROR] Chunked transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def load_asr(enable_punctuation: bool = True) -> Optional[ASRModule]:
    """
    Load and return an ASR model instance.
    """
    if not DEPENDENCIES_AVAILABLE:
        print("[WARNING] Cannot load ASR: required dependencies are not installed.")
        return None
    try:
        return ASRModule(enable_punctuation=enable_punctuation)
    except Exception as e:
        print(f"[ERROR] ASR model loading failed during instantiation: {e}")
        return None


if __name__ == "__main__":
    print("🧪 Testing ASR Module with Punctuation Restoration")
    print("=" * 50)

    if not DEPENDENCIES_AVAILABLE:
        print("[ERROR] ASR dependencies not available. Exiting.")
        exit(1)

    if torch.cuda.is_available():
        print(f"[OK] CUDA available: {torch.version.cuda}")
        print(f"📱 GPU: {torch.cuda.get_device_name()}")
    else:
        print("[WARNING] CUDA not available, using CPU.")

    if PUNCTUATION_AVAILABLE:
        print("[OK] Punctuation restoration is available.")
    else:
        print("[WARNING] Punctuation restoration is not available.")

    asr_model = load_asr(enable_punctuation=True)

    if asr_model and asr_model.is_available():
        print("[OK] ASR model loaded successfully!")
        print(f"[CLIPBOARD] Model ID: {asr_model.model_id}")
        print(f"📱 Device: {asr_model.device}")
        print(f"[MEMO] Punctuation enabled: {asr_model.is_punctuation_available()}")

        # Test with a sample audio file if it exists
        test_file = "demo.wav"
        try:
            import os
            if os.path.exists(test_file):
                print(f"\n[MUSIC] Testing with {test_file}...")
                transcription_result = asr_model.transcribe(test_file)
                if transcription_result:
                    print("\n--- Transcription Result ---")
                    print(f"  Raw Text: {transcription_result['text']}")
                    print(
                        "  Punctuated: "
                        f"{transcription_result['punctuated_text']}"
                    )
                    print(
                        f"  Time (ASR): {transcription_result['processing_time']}s"
                    )
                    print(
                        "  Time (Punc): "
                        f"{transcription_result['punctuation_time']}s"
                    )
                    print("--------------------------")
            else:
                print(f"\n[WARNING] Test file '{test_file}' not found. Skipping transcribe test.")
        except Exception as e:
            print(f"\n[ERROR] An error occurred during transcription test: {e}")