"""
Notes Generator V2 - Production Implementation for RTX 3050
Uses 4-bit quantization for memory efficiency

Model: ahmedhugging12/flan-t5-base-vtssum
Type: Seq2Seq (FLAN-T5-base fine-tuned)
Expected Memory: ~500MB (4-bit) vs ~1GB (FP16)
"""

import os
import gc
import time
import logging
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from nltk.tokenize import sent_tokenize

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class NotesGeneratorV2:
    """
    Production Notes Generator using 4-bit quantization.

    Optimized for RTX 3050 (4GB VRAM)
    """

    def __init__(
        self,
        model_name: str = "ahmedhugging12/flan-t5-base-vtssum",
        use_4bit: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Notes Generator with 4-bit quantization.

        Args:
            model_name: HuggingFace model identifier
            use_4bit: Use 4-bit quantization (recommended for 4GB VRAM)
            cache_dir: Optional cache directory
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), ".model_cache")

        # Model components
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False

        # Generation parameters (optimized for fine-tuned model)
        self.max_input_length = 512
        self.max_output_length = 25  # Model trained for ~15 token outputs
        self.chunk_size = 150  # Words per chunk
        self.overlap_size = 30  # Overlap for context

        logger.info(f"[NotesV2] Initializing with model: {model_name}")

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_model(self):
        """Load model with optional 4-bit quantization."""

        if self.model_loaded:
            logger.info("[NotesV2] Model already loaded")
            return

        logger.info(f"[NotesV2] Loading model (4-bit: {self.use_4bit})...")
        start_time = time.time()

        # Check GPU
        if not torch.cuda.is_available():
            logger.warning("[NotesV2] CUDA not available - using CPU")
            self.use_4bit = False  # Can't use 4-bit on CPU

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )

        # Load model with optional quantization
        if self.use_4bit and torch.cuda.is_available():
            # 4-bit quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=self.cache_dir
            )
        else:
            # FP16 or CPU
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                cache_dir=self.cache_dir
            )

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.model_loaded = True

        load_time = time.time() - start_time

        # Get memory info
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"[NotesV2] GPU Memory: {allocated:.2f}GB / {total:.2f}GB")

        logger.info(f"[NotesV2] Model loaded in {load_time:.2f}s")
        logger.info(f"[NotesV2] Device: {self.device}")

    def chunk_transcript(self, text: str) -> List[str]:
        """
        Chunk transcript with overlap for better context.

        Args:
            text: Lecture transcript

        Returns:
            List of text chunks
        """
        sentences = sent_tokenize(text)
        if not sentences:
            return [text]

        chunks = []
        current_chunk = []
        current_tokens = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))

            # Start new chunk if exceeds limit
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))

                # Overlap: keep last 2 sentences
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk[-1:]
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(len(self.tokenizer.encode(s, add_special_tokens=False))
                                   for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

            i += 1

        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        logger.info(f"[NotesV2] Created {len(chunks)} chunks from transcript")
        return chunks

    def generate_bullet_point(self, chunk: str) -> str:
        """
        Generate a single bullet point from a chunk.

        Args:
            chunk: Text chunk

        Returns:
            Bullet point summary
        """
        # Prepare input (model trained with specific format)
        prompt = f"summarize: {chunk}"

        inputs = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return summary.strip()

    def generate_notes(
        self,
        transcript: str,
        title: str = "Lecture Notes"
    ) -> Dict[str, Any]:
        """
        Generate bullet point notes from lecture transcript.

        Args:
            transcript: Lecture transcript text
            title: Optional title for notes

        Returns:
            Dictionary with notes and metadata
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded - call load_model() first")

        start_time = time.time()

        # Chunk transcript
        chunks = self.chunk_transcript(transcript)

        # Generate bullet points
        bullet_points = []

        logger.info(f"[NotesV2] Generating notes from {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            try:
                bullet = self.generate_bullet_point(chunk)
                if bullet and len(bullet) > 5:  # Filter out very short/empty
                    bullet_points.append(bullet)
                    logger.info(f"[NotesV2] Chunk {i+1}/{len(chunks)}: {bullet[:50]}...")
            except Exception as e:
                logger.error(f"[NotesV2] Error processing chunk {i+1}: {e}")
                continue

        total_time = time.time() - start_time

        # Format as markdown
        markdown = f"# {title}\n\n"
        for i, bullet in enumerate(bullet_points, 1):
            markdown += f"{i}. {bullet}\n"

        return {
            "title": title,
            "bullet_points": bullet_points,
            "markdown": markdown,
            "metadata": {
                "num_bullets": len(bullet_points),
                "num_chunks": len(chunks),
                "generation_time": round(total_time, 2),
                "model": self.model_name,
                "success": len(bullet_points) > 0
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status."""
        info = {
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "device": str(self.device) if self.device else None,
            "use_4bit": self.use_4bit,
        }

        if torch.cuda.is_available() and self.model_loaded:
            info["gpu_memory_allocated_gb"] = round(
                torch.cuda.memory_allocated(0) / (1024**3), 2
            )
            info["gpu_memory_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )

        return info

    def cleanup(self):
        """Clean up resources."""
        logger.info("[NotesV2] Cleaning up resources...")

        self.model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        self.model_loaded = False
        logger.info("[NotesV2] Cleanup complete")


# Factory function
def create_notes_generator(
    model_name: str = "ahmedhugging12/flan-t5-base-vtssum",
    use_4bit: bool = True
) -> NotesGeneratorV2:
    """
    Create and initialize Notes Generator V2.

    Args:
        model_name: HuggingFace model identifier
        use_4bit: Use 4-bit quantization

    Returns:
        Initialized NotesGeneratorV2 instance
    """
    generator = NotesGeneratorV2(model_name=model_name, use_4bit=use_4bit)
    generator.load_model()
    return generator


# Test function
def test_notes_generator():
    """Test Notes Generator V2 with sample lecture."""
    logger.info("="*80)
    logger.info("Testing Notes Generator V2")
    logger.info("="*80)

    # Sample lecture
    lecture = """
    Today we'll discuss operant conditioning, a learning theory developed by
    B.F. Skinner. Operant conditioning involves learning through consequences.
    When a behavior is followed by a reward, it is more likely to be repeated.
    This is called reinforcement. When a behavior is followed by punishment,
    it is less likely to be repeated. Skinner demonstrated these principles
    using a device called the Skinner box, where animals could press a lever
    to receive food rewards. There are two types of reinforcement: positive
    reinforcement, which adds something desirable, and negative reinforcement,
    which removes something undesirable. Both increase the likelihood of a
    behavior. Similarly, there are two types of punishment: positive punishment
    adds something undesirable, and negative punishment removes something desirable.
    Both decrease the likelihood of a behavior.
    """

    # Create generator
    generator = create_notes_generator()

    # Print model info
    info = generator.get_model_info()
    logger.info(f"\nModel Info: {info}")

    # Generate notes
    result = generator.generate_notes(
        transcript=lecture,
        title="Operant Conditioning"
    )

    # Print results
    logger.info("\n" + "="*80)
    logger.info("Generated Notes:")
    logger.info("="*80)
    logger.info("\n" + result["markdown"])
    logger.info("="*80)
    logger.info(f"Metadata: {result['metadata']}")
    logger.info("="*80)

    # Cleanup
    generator.cleanup()

    return result


if __name__ == "__main__":
    test_notes_generator()
