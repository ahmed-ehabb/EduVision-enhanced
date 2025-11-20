"""
Quiz Generator V2 - Production Implementation
Uses the tested and validated model configuration for RTX 3050 (4GB VRAM)

Based on successful testing from PHASE1_QUIZ_MODEL_RESULTS.md
- Model: ahmedhugging12/Llama-3.2-3B-Psychology-Merged
- Quantization: 4-bit NF4
- Memory: ~2GB VRAM
- Speed: ~12 tok/s
- Format: Llama 3.2 chat template (CRITICAL!)
"""

import os
import gc
import time
import logging
import re
from typing import List, Dict, Optional, Any
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class QuizGeneratorV2:
    """
    Production Quiz Generator using tested configuration.

    Key Features:
    - 4-bit quantization for RTX 3050 (4GB VRAM)
    - Llama 3.2 chat template (correct format)
    - RAG integration with FAISS vector store
    - Memory-efficient implementation
    """

    def __init__(
        self,
        model_name: str = "ahmedhugging12/Llama-3.2-3B-Psychology-Merged",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Quiz Generator with tested configuration.

        Args:
            model_name: HuggingFace model identifier (merged model)
            cache_dir: Optional cache directory for models
        """
        self.model_name = model_name
        # Use project .model_cache directory where model is already stored
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), ".model_cache")

        # Model components
        self.model = None
        self.tokenizer = None
        self.vector_store = None
        self.embedding_model = None

        # State tracking
        self.model_loaded = False
        self.device = None

        logger.info(f"[QuizV2] Initializing with model: {model_name}")

    def load_model(self):
        """Load model with 4-bit quantization (tested configuration)."""

        if self.model_loaded:
            logger.info("[QuizV2] Model already loaded")
            return

        logger.info("[QuizV2] Loading model with 4-bit quantization...")
        start_time = time.time()

        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - GPU required for this model")

        # CRITICAL: Clear GPU memory before loading large model
        # This prevents ACCESS_VIOLATION crashes when previous models consumed memory
        logger.info("[QuizV2] Clearing GPU cache before loading...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            mem_before = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"[QuizV2] GPU memory before load: {mem_before:.2f} GB")

        # 4-bit quantization config (tested and validated)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # Load model with tested config
        # Use max_memory to be more conservative when loading after other models
        try:
            logger.info("[QuizV2] Starting model load from HuggingFace...")
            logger.info(f"[QuizV2] Model path: {self.model_name}")
            logger.info(f"[QuizV2] Cache dir: {self.cache_dir}")

            # Try loading WITHOUT max_memory constraint first (simpler, might avoid issues)
            logger.info("[QuizV2] Calling AutoModelForCausalLM.from_pretrained()...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                low_cpu_mem_usage=True
            )
            logger.info("[QuizV2] Model loaded successfully from from_pretrained()")
        except Exception as e:
            logger.warning(f"[QuizV2] First load attempt failed: {e}")
            logger.info("[QuizV2] Retrying with even more conservative settings...")

            # Clear memory again
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Second attempt: even more conservative
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory={0: "2.5GB"},  # Even more conservative
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                low_cpu_mem_usage=True
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
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
            logger.info(f"[QuizV2] GPU Memory: {allocated:.2f}GB / {total:.2f}GB")

        logger.info(f"[QuizV2] Model loaded in {load_time:.2f}s")
        logger.info(f"[QuizV2] Device: {self.device}")

    def load_rag_components(self, pdf_path: Optional[str] = None):
        """
        Load RAG components (vector store + embeddings).

        Args:
            pdf_path: Path to Psychology textbook PDF (optional)
        """
        try:
            from sentence_transformers import SentenceTransformer
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from PyPDF2 import PdfReader
        except ImportError as e:
            logger.error(f"[QuizV2] RAG dependencies not installed: {e}")
            return

        logger.info("[QuizV2] Loading RAG components...")

        # Load embedding model (CPU to save GPU memory)
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device='cpu'
        )
        logger.info("[QuizV2] Embedding model loaded on CPU")

        # Check for existing vector store
        index_path = os.path.join(self.cache_dir, "psych_book_faiss_index")

        if os.path.exists(index_path):
            logger.info("[QuizV2] Loading existing vector store...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.vector_store = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("[QuizV2] Vector store loaded from cache")
            return

        # Build new vector store if PDF provided
        if pdf_path and os.path.exists(pdf_path):
            logger.info(f"[QuizV2] Building vector store from PDF: {pdf_path}")

            # Extract text from PDF
            reader = PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text()

            cleaned_text = re.sub(r'\s+', ' ', full_text)
            logger.info(f"[QuizV2] Extracted {len(reader.pages)} pages, {len(cleaned_text)} characters")

            # Split into chapters and chunks
            chapter_pattern = r'Chapter \d+'
            chapters = re.split(chapter_pattern, cleaned_text)
            chapter_titles = re.findall(chapter_pattern, cleaned_text)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            all_chunks = []
            all_metadata = []

            for chapter_title, chapter_text in zip(chapter_titles, chapters[1:]):
                chunks = text_splitter.split_text(chapter_text)
                all_chunks.extend(chunks)
                all_metadata.extend([{"chapter_title": chapter_title} for _ in chunks])

            logger.info(f"[QuizV2] Created {len(all_chunks)} chunks")

            # Build FAISS vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            self.vector_store = FAISS.from_texts(
                all_chunks,
                embedding=embeddings,
                metadatas=all_metadata
            )

            # Save for future use
            self.vector_store.save_local(index_path)
            logger.info(f"[QuizV2] Vector store saved to {index_path}")
        else:
            logger.warning("[QuizV2] No PDF provided, using lightweight knowledge base")
            self._create_lightweight_knowledge_base()

    def _create_lightweight_knowledge_base(self):
        """Create lightweight psychology knowledge base."""
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        except ImportError:
            logger.warning("[QuizV2] Cannot create knowledge base - dependencies missing")
            return

        # Curated psychology concepts
        psychology_knowledge = [
            "Psychology is the scientific study of mind and behavior using empirical methods.",
            "Classical conditioning involves learning associations between neutral and meaningful stimuli.",
            "Operant conditioning uses reinforcement and punishment to shape behavior over time.",
            "Cognitive psychology studies mental processes including memory, attention, and perception.",
            "Social psychology examines how people think about and influence each other.",
            "Developmental psychology studies changes in behavior and abilities across the lifespan.",
            "Memory involves encoding, storing, and retrieving information from experience.",
            "Learning is a relatively permanent change in behavior resulting from experience.",
            "Motivation includes internal drives and external incentives that direct behavior.",
            "Personality refers to consistent patterns of thinking, feeling, and behaving.",
        ]

        metadata = [{"source": "psychology_concepts", "chunk_id": i} for i in range(len(psychology_knowledge))]

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        self.vector_store = FAISS.from_texts(
            psychology_knowledge,
            embedding=embeddings,
            metadatas=metadata
        )

        logger.info(f"[QuizV2] Lightweight knowledge base created with {len(psychology_knowledge)} concepts")

    def retrieve_context(self, lecture_text: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant context from vector store based on lecture.

        Uses lecture-based retrieval (tested approach).

        Args:
            lecture_text: Lecture transcript text
            top_k: Number of top chunks to retrieve

        Returns:
            List of relevant context chunks
        """
        if not self.vector_store or not self.embedding_model:
            logger.warning("[QuizV2] Vector store not available")
            return []

        # Split lecture into chunks
        query_chunks = [lecture_text]
        if len(lecture_text) > 1000:
            paras = lecture_text.split("\n\n")
            query_chunks = [para.strip() for para in paras if para.strip()]

            # Further split long paragraphs
            refined_chunks = []
            for para in query_chunks:
                if len(para) > 500:
                    sentences = para.split('. ')
                    half = len(sentences)//2
                    refined_chunks.append('. '.join(sentences[:half]) + '.')
                    refined_chunks.append('. '.join(sentences[half:]) + '.')
                else:
                    refined_chunks.append(para)
            query_chunks = refined_chunks

        # Retrieve relevant chunks for each lecture chunk
        retrieved_docs = []
        for q in query_chunks:
            q_emb = self.embedding_model.encode(q)
            docs = self.vector_store.similarity_search_by_vector(q_emb, k=2)
            retrieved_docs.extend(docs)

        # Deduplicate
        unique_contexts = []
        seen_texts = set()
        for doc in retrieved_docs:
            text = doc.page_content
            if text not in seen_texts:
                unique_contexts.append(text)
                seen_texts.add(text)

        # Return top K
        context_chunks = unique_contexts[:top_k]
        logger.info(f"[QuizV2] Retrieved {len(context_chunks)} relevant chunks")

        return context_chunks

    def generate_mcq_questions(
        self,
        context: str,
        num_questions: int = 3,
        max_new_tokens: int = 500,
        temperature: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Generate MCQ questions using CORRECT Llama 3.2 format.

        Args:
            context: Educational content (from RAG retrieval)
            num_questions: Number of questions to generate
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            List of question dictionaries with structure:
            [
                {
                    "question": str,
                    "options": {"a": str, "b": str, "c": str, "d": str},
                    "correct_answer": str,
                    "raw_text": str
                },
                ...
            ]
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded - call load_model() first")

        # Build prompt with clear formatting instructions
        prompt = (
            f"The following context is from a psychology textbook and lecture:\n{context}\n\n"
            f"[Task]\nYou are an educational AI assisting in quiz creation. "
            f"Based on the above content, generate {num_questions} multiple-choice questions. "
            "Each question should have four options (a, b, c, d) and indicate the correct answer. "
            "Ensure the questions focus on important details from the content.\n\n"
            "Format each question exactly as follows:\n"
            "Question: [question text]\n"
            "a) [option a]\n"
            "b) [option b]\n"
            "c) [option c]\n"
            "d) [option d]\n"
            "Correct answer: [letter]\n\n"
            "Separate each question with a blank line."
        )

        # Generate with CORRECT Llama 3.2 format (CRITICAL!)
        raw_output = self._generate(prompt, max_new_tokens, temperature)

        # Log raw output for debugging
        logger.info(f"[QuizV2] Raw output length: {len(raw_output)} chars")
        logger.info(f"[QuizV2] Raw output (first 800 chars):\n{raw_output[:800]}")
        if len(raw_output) > 800:
            logger.info(f"[QuizV2] Raw output (last 400 chars):\n{raw_output[-400:]}")

        # Parse the generated text into structured questions
        parsed_questions = self._parse_mcq_output(raw_output)

        return parsed_questions

    def generate_open_ended_questions(
        self,
        context: str,
        num_questions: int = 3,
        question_type: str = "short_answer",
        include_answers: bool = True,
        max_new_tokens: int = 1000,
        temperature: float = 0.3
    ) -> str:
        """
        Generate open-ended questions (short answer, essay, or discussion).

        Args:
            context: Educational content (from RAG retrieval or lecture)
            num_questions: Number of questions to generate
            question_type: Type of questions - "short_answer", "essay", or "discussion"
            include_answers: If True, generate sample answers/rubrics for each question
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (higher for more creative questions)

        Returns:
            Generated open-ended questions as text (with optional answers/rubrics)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded - call load_model() first")

        # Customize prompt based on question type
        type_instructions = {
            "short_answer": (
                "Generate short-answer questions that require 2-3 sentence responses. "
                "Focus on conceptual understanding and application of key ideas."
            ),
            "essay": (
                "Generate essay questions that require detailed, multi-paragraph responses. "
                "Focus on analysis, synthesis, and critical thinking about major concepts."
            ),
            "discussion": (
                "Generate discussion questions that encourage critical thinking and debate. "
                "Focus on thought-provoking topics that connect concepts to real-world scenarios."
            )
        }

        instructions = type_instructions.get(
            question_type,
            type_instructions["short_answer"]
        )

        # Build prompt with or without answers
        if include_answers:
            prompt = (
                f"The following context is from a psychology textbook and lecture:\n{context}\n\n"
                f"[Task]\nYou are an educational AI assisting in assessment creation. "
                f"Based on the above content, generate {num_questions} open-ended questions WITH sample answers.\n\n"
                f"{instructions}\n\n"
                "Format each question and answer as follows:\n\n"
                "Question 1: [question text]\n"
                "Sample Answer: [provide a comprehensive sample answer that demonstrates the expected level of understanding]\n\n"
                "Ensure questions:\n"
                "- Assess deep understanding, not just memorization\n"
                "- Are specific to the content provided\n"
                "- Encourage critical thinking and application\n"
                "- Are appropriately challenging for college-level students"
            )
        else:
            prompt = (
                f"The following context is from a psychology textbook and lecture:\n{context}\n\n"
                f"[Task]\nYou are an educational AI assisting in assessment creation. "
                f"Based on the above content, generate {num_questions} open-ended questions.\n\n"
                f"{instructions}\n\n"
                "Format each question clearly, numbered (1, 2, 3...), and ensure they:\n"
                "- Assess deep understanding, not just memorization\n"
                "- Are specific to the content provided\n"
                "- Encourage critical thinking and application\n"
                "- Are appropriately challenging for college-level students"
            )

        # Generate with CORRECT Llama 3.2 format (CRITICAL!)
        result = self._generate(prompt, max_new_tokens, temperature)

        return result

    def _generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """
        Generate response using CORRECT Llama 3.2 format.

        CRITICAL: Must use tokenizer.apply_chat_template() for Llama 3.2!
        """
        # Format as chat message (Llama 3.2 format)
        messages = [{"role": "user", "content": prompt}]

        # Apply Llama 3.2 chat template (CRITICAL!)
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Generate
        logger.info(f"[QuizV2] Generating {max_new_tokens} tokens...")
        start_time = time.time()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,
                top_p=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        gen_time = time.time() - start_time

        # Extract only new tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        logger.info(f"[QuizV2] Generated in {gen_time:.2f}s (~{max_new_tokens/gen_time:.1f} tok/s)")

        return output_text.strip()

    def _parse_mcq_output(self, raw_output: str) -> List[Dict[str, Any]]:
        """
        Parse generated MCQ text into structured question dictionaries.

        Args:
            raw_output: Raw text output from model

        Returns:
            List of parsed question dictionaries
        """
        questions = []

        # Split by "Question:" markers
        question_blocks = re.split(r'(?=Question:)', raw_output)

        for block in question_blocks:
            if not block.strip() or 'Question:' not in block:
                continue

            try:
                # Extract question text - everything after "Question:" until we hit "a)"
                q_match = re.search(r'Question:\s*(.+?)(?=\na\))', block, re.DOTALL)
                if not q_match:
                    logger.warning(f"[QuizV2] Could not find question text in block")
                    continue
                question_text = q_match.group(1).strip()

                # Extract options
                options = {}
                for letter in ['a', 'b', 'c', 'd']:
                    # Match option text until next option or "Correct answer:" line
                    opt_pattern = rf'{letter}\)\s*(.+?)(?=\n[a-d]\)|Correct answer:|\Z)'
                    opt_match = re.search(opt_pattern, block, re.DOTALL | re.IGNORECASE)
                    if opt_match:
                        opt_text = opt_match.group(1).strip()
                        # Remove trailing newlines
                        opt_text = opt_text.split('\n')[0].strip()
                        options[letter] = opt_text

                # Extract correct answer
                ans_match = re.search(r'Correct answer:\s*([a-d])', block, re.IGNORECASE)
                correct_answer = ans_match.group(1).lower() if ans_match else ''

                # Only add if we have a question and all 4 options
                if question_text and len(options) == 4 and correct_answer:
                    questions.append({
                        "question": question_text,
                        "options": options,
                        "correct_answer": correct_answer,
                        "raw_text": block.strip()
                    })
                    logger.info(f"[QuizV2] Successfully parsed question: {question_text[:50]}...")
                else:
                    logger.warning(f"[QuizV2] Incomplete question - text: {bool(question_text)}, options: {len(options)}/4, answer: {bool(correct_answer)}")

            except Exception as e:
                logger.warning(f"[QuizV2] Failed to parse question block: {e}")
                continue

        # If parsing failed, return a placeholder with the raw text
        if not questions and raw_output.strip():
            logger.warning("[QuizV2] Could not parse any questions, returning raw text as placeholder")
            questions.append({
                "question": "Generated content (parsing failed)",
                "options": {"a": "See raw output", "b": "", "c": "", "d": ""},
                "correct_answer": "a",
                "raw_text": raw_output[:500]  # First 500 chars
            })

        logger.info(f"[QuizV2] Parsed {len(questions)} questions from output")
        return questions

    def generate_quiz_from_lecture(
        self,
        lecture_text: str,
        num_questions: int = 5,
        max_new_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Full pipeline: Retrieve context + Generate MCQ questions.

        Args:
            lecture_text: Lecture transcript
            num_questions: Number of questions
            max_new_tokens: Max tokens to generate

        Returns:
            Dictionary with questions and metadata
        """
        start_time = time.time()

        # 1. Retrieve context
        context_chunks = self.retrieve_context(lecture_text, top_k=5)
        context_text = "\n\n".join(context_chunks)

        # 2. Generate MCQ questions (already parsed into structured format)
        questions = self.generate_mcq_questions(
            context=context_text,
            num_questions=num_questions,
            max_new_tokens=max_new_tokens
        )

        total_time = time.time() - start_time

        return {
            "questions": questions,
            "metadata": {
                "num_questions_requested": num_questions,
                "num_questions_generated": len(questions),
                "generation_time": round(total_time, 2),
                "model": self.model_name,
                "context_chunks": len(context_chunks),
                "success": len(questions) > 0
            }
        }


    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status."""
        info = {
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "device": str(self.device) if self.device else None,
            "vector_store_loaded": self.vector_store is not None,
            "embedding_model_loaded": self.embedding_model is not None,
        }

        if torch.cuda.is_available() and self.model_loaded:
            info["gpu_memory_allocated_gb"] = round(
                torch.cuda.memory_allocated(0) / (1024**3), 2
            )
            info["gpu_memory_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )

        return info

    def unload_model(self):
        """Unload model and free GPU memory (alias for cleanup)."""
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        logger.info("[QuizV2] Cleaning up resources...")

        self.model = None
        self.tokenizer = None
        self.embedding_model = None
        self.vector_store = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        self.model_loaded = False
        logger.info("[QuizV2] Cleanup complete")


# Factory function
def create_quiz_generator(
    model_name: str = "ahmedhugging12/Llama-3.2-3B-Psychology-Merged",
    pdf_path: Optional[str] = None
) -> QuizGeneratorV2:
    """
    Create and initialize Quiz Generator V2.

    Args:
        model_name: HuggingFace model identifier
        pdf_path: Optional path to Psychology PDF

    Returns:
        Initialized QuizGeneratorV2 instance
    """
    generator = QuizGeneratorV2(model_name=model_name)
    generator.load_model()
    generator.load_rag_components(pdf_path=pdf_path)
    return generator


# Test function
def test_quiz_generator():
    """Test Quiz Generator V2 with sample lecture."""
    logger.info("="*80)
    logger.info("Testing Quiz Generator V2")
    logger.info("="*80)

    # Sample lecture
    lecture = """
    Today we'll discuss classical conditioning, discovered by Ivan Pavlov.
    Classical conditioning is a learning process where a neutral stimulus becomes
    associated with a meaningful stimulus. In Pavlov's famous experiment, dogs
    learned to salivate at the sound of a bell that was paired with food.
    """

    # Create generator
    generator = create_quiz_generator()

    # Print model info
    info = generator.get_model_info()
    logger.info(f"Model Info: {info}")

    # Generate quiz
    result = generator.generate_quiz_from_lecture(
        lecture_text=lecture,
        num_questions=3
    )

    # Print results
    logger.info("\n" + "="*80)
    logger.info("Generated Questions:")
    logger.info("="*80)

    for i, q in enumerate(result["questions"], 1):
        logger.info(f"\n{i}. {q['question']}")
        for opt in q['options']:
            logger.info(f"   {opt}")
        logger.info(f"   Correct: {q['correct_answer']}")

    logger.info("\n" + "="*80)
    logger.info(f"Metadata: {result['metadata']}")
    logger.info("="*80)

    # Cleanup
    generator.cleanup()

    return result


if __name__ == "__main__":
    test_quiz_generator()
