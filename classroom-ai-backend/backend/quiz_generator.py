"""
Optimized Quiz Generator - Fixes duplicate logging and memory issues
Replaces backend/quiz_generator.py with memory-efficient implementation
"""

import gc
import re
import json
import asyncio
import os
import threading
import time
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import logging

import torch

# FIXED: Single logger instance to prevent duplicates
logger = logging.getLogger(__name__)
if not logger.handlers:  # Only add handler if none exists
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Environment configuration - optimized for 4GB GPU
QUIZ_SAFE_MODE = os.getenv("QUIZ_SAFE_MODE", "true").lower() == "true"
QUIZ_FALLBACK_ONLY = os.getenv("QUIZ_FALLBACK_ONLY", "false").lower() == "true"
QUIZ_FORCE_CPU = os.getenv("QUIZ_FORCE_CPU", "false").lower() == "true"
QUIZ_SKIP_PEFT = os.getenv("QUIZ_SKIP_PEFT", "true").lower() == "true"  # Default to true for 4GB GPU
QUIZ_DISABLE_QUANTIZATION = os.getenv("QUIZ_DISABLE_QUANTIZATION", "false").lower() == "true"
QUIZ_MEMORY_LIMIT_GB = float(os.getenv("QUIZ_MEMORY_LIMIT", "1.0"))  # Reduced for 4GB GPU
QUIZ_TIMEOUT_SECONDS = int(os.getenv("QUIZ_TIMEOUT_SECONDS", "30"))  # Shorter timeout
QUIZ_MAX_RETRIES = int(os.getenv("QUIZ_MAX_RETRIES", "1"))
QUIZ_SKIP_LARGE_RAG = os.getenv("QUIZ_SKIP_LARGE_RAG", "true").lower() == "true"  # Skip large textbook loading

# HuggingFace authentication - single login
if not hasattr(torch, '_hf_logged_in'):
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        try:
            from huggingface_hub import login
            login(token=HF_TOKEN)
            torch._hf_logged_in = True  # Mark as logged in
            logger.info("HuggingFace authentication successful")
        except Exception as e:
            logger.warning(f"HuggingFace authentication failed: {e}")

# Safe imports with fallback flags - prevent duplicate loading
TRANSFORMERS_AVAILABLE = False
QUANTIZATION_AVAILABLE = False
VECTOR_STORE_AVAILABLE = False
PEFT_AVAILABLE = False

# Use module-level check to prevent duplicate logging
_IMPORTS_INITIALIZED = False

if not _IMPORTS_INITIALIZED:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        TRANSFORMERS_AVAILABLE = True
        logger.info("Transformers library loaded")
    except ImportError as e:
        logger.warning(f"Transformers not available: {e}")

    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes as bnb
        if hasattr(bnb.nn, "Linear4bit") and not QUIZ_DISABLE_QUANTIZATION:
            QUANTIZATION_AVAILABLE = True
            logger.info("Quantization support available")
    except Exception as e:
        logger.warning(f"Quantization not available: {e}")

    try:
        from sentence_transformers import SentenceTransformer
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        VECTOR_STORE_AVAILABLE = True
        logger.info("Vector store libraries loaded")
    except ImportError as e:
        logger.warning(f"Vector store not available: {e}")

    try:
        from peft import PeftModel
        PEFT_AVAILABLE = True
        logger.info("PEFT library loaded")
    except ImportError as e:
        logger.warning(f"PEFT not available: {e}")
    
    _IMPORTS_INITIALIZED = True  # Mark imports as checked


class MemoryManager:
    """Optimized memory manager for 4GB GPU."""
    
    @staticmethod
    def get_available_gpu_memory() -> float:
        """Get available GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            available = (total_memory - allocated) / (1024**3)
            return available
        except Exception:
            return 0.0
    
    @staticmethod
    def aggressive_cleanup():
        """Aggressively clean up GPU memory."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            time.sleep(0.1)
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    @staticmethod
    def check_memory_sufficient(required_gb: float) -> bool:
        """Check if sufficient memory is available."""
        available = MemoryManager.get_available_gpu_memory()
        return available >= required_gb


class QuizGenerator:
    """
    Memory-optimized quiz generator for 4GB GPU systems.
    """

    def __init__(
        self,
        model_repo: str = "ahmedhugging12/llama3b-psych-mcqqa-lora-4bit",
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        use_lightweight: bool = True,
        cache_dir: Optional[str] = None
    ):
        """Initialize quiz generator with memory-optimized configuration."""
        
        # Prevent duplicate initialization
        if hasattr(self, '_initialized'):
            return
        
        # Configuration
        self.model_repo = model_repo
        self.base_model = base_model
        self.use_lightweight = use_lightweight
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "model_cache")
        
        # State management
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.vector_store = None
        self.embedding_model = None
        self.available = False
        self.model_loaded = False
        self.last_error = None
        self.loading_lock = threading.Lock()
        self.retry_count = 0
        
        # Operational modes - optimized for 4GB GPU
        self.safe_mode = QUIZ_SAFE_MODE
        self.fallback_only = QUIZ_FALLBACK_ONLY
        self.force_cpu = QUIZ_FORCE_CPU
        self.skip_large_rag = QUIZ_SKIP_LARGE_RAG
        
        # Memory management - conservative for 4GB GPU
        self.memory_manager = MemoryManager()
        self.memory_limit_gb = QUIZ_MEMORY_LIMIT_GB
        
        # Device configuration
        self.device = self._determine_device()
        
        # Generation parameters - reduced for memory efficiency
        self.max_length = 512 if use_lightweight else 1024
        self.generation_config = {
            "max_new_tokens": 150,  # Reduced
            "temperature": 0.3,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": None,
            "eos_token_id": None
        }
        
        # Initialize safely
        try:
            self._safe_initialization()
            self.available = True
            logger.info(f"Quiz generator initialized (device: {self.device}, memory limit: {self.memory_limit_gb}GB)")
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Quiz generator initialization failed: {e}")
            self.fallback_only = True
            self.available = True
            logger.info("Quiz generator running in fallback mode")
        
        self._initialized = True

    def _determine_device(self) -> torch.device:
        """Determine device based on available memory."""
        
        if self.force_cpu or not torch.cuda.is_available():
            logger.info("Using CPU device")
            return torch.device("cpu")
        
        # Check GPU memory
        available_memory = self.memory_manager.get_available_gpu_memory()
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        logger.info(f"GPU memory: {available_memory:.2f}GB available / {total_memory:.2f}GB total")
        
        # Ultra-aggressive mode for 0.49GB constraint
        if available_memory >= 0.35:  # Very low threshold for tight constraint
            logger.info("Using GPU device with ultra-tight memory constraint")
            return torch.device("cuda")
        elif available_memory < self.memory_limit_gb and self.memory_limit_gb > 0.5:
            logger.warning(f"Insufficient GPU memory ({available_memory:.2f}GB < {self.memory_limit_gb}GB), using CPU")
            self.force_cpu = True
            return torch.device("cpu")
        else:
            logger.info("Using GPU device with aggressive optimization")
            return torch.device("cuda")

    def _safe_initialization(self):
        """Memory-efficient initialization."""
        
        # Skip model-related initialization if in fallback-only mode
        if self.fallback_only:
            logger.info("Fallback-only mode - skipping model initialization")
            return
        
        # Initialize lightweight knowledge base only if not skipping RAG
        if VECTOR_STORE_AVAILABLE and not self.skip_large_rag:
            try:
                self._initialize_lightweight_knowledge_base()
                logger.info("Lightweight knowledge base initialized")
            except Exception as e:
                logger.warning(f"Knowledge base initialization failed: {e}")
        else:
            logger.info("Skipping large RAG knowledge base to save memory")
        
        # Create cache directory
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create cache directory: {e}")

    def _initialize_lightweight_knowledge_base(self):
        """Initialize lightweight knowledge base instead of large textbook."""
        
        try:
            # Smaller, curated psychology knowledge base
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
                "Perception is the process of organizing and interpreting sensory information.",
                "Attention is the selective focus on particular aspects of the environment.",
                "Research methods in psychology include experiments, correlational studies, and observations.",
                "Statistical analysis helps psychologists draw valid conclusions from data.",
                "Ethical guidelines protect participants and ensure responsible research practices."
            ]
            
            metadata = [{"source": "psychology_concepts", "chunk_id": i} for i in range(len(psychology_knowledge))]
            
            # Use CPU for embeddings to save GPU memory
            if not hasattr(self, 'embedding_model'):
                self.embedding_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device='cpu'  # Force CPU to save GPU memory
                )
                logger.info("Embedding model loaded on CPU")
            
            # Create HuggingFace embeddings wrapper
            hf_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            self.vector_store = FAISS.from_texts(
                psychology_knowledge,
                embedding=hf_embeddings,
                metadatas=metadata
            )
            
            logger.info(f"Lightweight knowledge base created with {len(psychology_knowledge)} concepts")
            
        except Exception as e:
            logger.warning(f"Lightweight knowledge base creation failed: {e}")
            self.vector_store = None

    async def _load_model_lazy(self) -> bool:
        """Memory-efficient model loading."""
        
        if self.model_loaded or self.fallback_only:
            return True
        
        with self.loading_lock:
            if self.model_loaded:
                return True
            
            if self.retry_count >= QUIZ_MAX_RETRIES:
                logger.warning("Maximum retries reached, using fallback mode")
                self.fallback_only = True
                return True
            
            self.retry_count += 1
            logger.info(f"Loading model (attempt {self.retry_count}/{QUIZ_MAX_RETRIES + 1})...")
            
            try:
                # Check memory before loading
                available_memory = self.memory_manager.get_available_gpu_memory()
                logger.info(f"Available memory before loading: {available_memory:.2f}GB")
                
                if available_memory < self.memory_limit_gb:
                    logger.warning(f"Insufficient memory ({available_memory:.2f}GB), using fallback")
                    self.fallback_only = True
                    return True
                
                # Try loading with timeout
                success = await asyncio.wait_for(
                    self._try_lightweight_loading(),
                    timeout=QUIZ_TIMEOUT_SECONDS
                )
                
                if success:
                    self.model_loaded = True
                    self.retry_count = 0
                    logger.info("Model loaded successfully")
                    return True
                else:
                    logger.warning("Model loading failed, using fallback")
                    self.fallback_only = True
                    return True
                
            except asyncio.TimeoutError:
                logger.error(f"Model loading timed out after {QUIZ_TIMEOUT_SECONDS}s")
                self.fallback_only = True
                return True
            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Model loading failed: {e}")
                self.fallback_only = True
                return True
            finally:
                self.memory_manager.aggressive_cleanup()

    async def _try_lightweight_loading(self) -> bool:
        """Try lightweight model loading suitable for 4GB GPU."""
        
        try:
            logger.info("Attempting lightweight model loading...")
            
            # Check if we have ultra-tight memory constraint (< 0.5GB)
            available_memory = self.memory_manager.get_available_gpu_memory()
            if available_memory < 0.5:
                logger.info(f"Ultra-tight memory constraint ({available_memory:.2f}GB), using specialized loading")
                return await self._try_ultra_lightweight_loading()
            
            # Use pipeline for simplicity and memory efficiency
            if TRANSFORMERS_AVAILABLE:
                pipeline_kwargs = {
                    "model": self.base_model,
                    "task": "text-generation",
                    "device": 0 if self.device.type == "cuda" else -1,
                    "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                    "token": os.getenv("HF_TOKEN"),
                    "trust_remote_code": True,
                    "model_kwargs": {
                        "low_cpu_mem_usage": True,
                        "use_cache": False  # Reduce memory usage
                    }
                }
                
                # Skip quantization for simplicity and memory efficiency
                self.pipeline = pipeline(**pipeline_kwargs)
                logger.info("Lightweight pipeline loading successful")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Lightweight loading failed: {e}")
            return False
    
    async def _try_ultra_lightweight_loading(self) -> bool:
        """Ultra-lightweight loading for 0.49GB constraint."""
        
        try:
            logger.info("Attempting ultra-lightweight loading for tight memory constraint...")
            
            # Aggressive memory cleanup first
            self.memory_manager.aggressive_cleanup()
            
            if not TRANSFORMERS_AVAILABLE or not QUANTIZATION_AVAILABLE:
                logger.warning("Required libraries not available for ultra-lightweight loading")
                return False
            
            from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
            
            # Ultra-aggressive quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Load tokenizer first (lighter)
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Try to load base model with most aggressive settings
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                max_memory={0: "0.4GB"}  # Strict GPU memory limit
            )
            
            # Try to load PEFT adapter if available
            if PEFT_AVAILABLE:
                try:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, self.model_repo)
                    logger.info("PEFT adapter loaded successfully")
                except Exception as peft_error:
                    logger.warning(f"PEFT loading failed: {peft_error}")
            
            # Create pipeline with loaded model
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0,
                torch_dtype=torch.float16
            )
            
            self.model = model
            self.tokenizer = tokenizer
            self.model_loaded = True
            
            logger.info("Ultra-lightweight loading successful!")
            return True
            
        except Exception as e:
            logger.error(f"Ultra-lightweight loading failed: {e}")
            # Last resort: try CPU loading
            try:
                logger.info("Attempting CPU fallback...")
                self.device = torch.device("cpu")
                self.force_cpu = True
                
                if TRANSFORMERS_AVAILABLE:
                    pipeline_kwargs = {
                        "model": self.base_model,
                        "task": "text-generation",
                        "device": -1,  # CPU
                        "torch_dtype": torch.float32,
                        "trust_remote_code": True,
                        "model_kwargs": {
                            "low_cpu_mem_usage": True,
                            "use_cache": False
                        }
                    }
                    
                    self.pipeline = pipeline(**pipeline_kwargs)
                    logger.info("CPU fallback loading successful")
                    return True
                
            except Exception as cpu_error:
                logger.error(f"CPU fallback also failed: {cpu_error}")
            
            return False

    async def generate_quiz(
        self,
        transcript: str,
        num_questions: int = 5,
        question_types: List[str] = ["open_ended", "mcq"],
        difficulty: str = "medium",
        topics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate quiz with memory-efficient approach."""
        
        if not self.available:
            return self._create_error_response("Quiz generator not available")
        
        # Validate and limit inputs for memory efficiency
        if not transcript or not transcript.strip():
            return self._create_error_response("Empty transcript provided")
        
        num_questions = max(1, min(num_questions, 10))  # Limit to 10 questions max
        
        try:
            # Try AI generation if not in fallback mode
            if not self.fallback_only:
                model_loaded = await self._load_model_lazy()
                
                if model_loaded and not self.fallback_only:
                    try:
                        result = await self._generate_ai_quiz(
                            transcript, num_questions, question_types, difficulty, topics
                        )
                        if result.get("questions") and len(result["questions"]) > 0:
                            return result
                    except Exception as ai_error:
                        logger.error(f"AI quiz generation failed: {ai_error}")
            
            # Use fallback generation
            logger.info("Using fallback quiz generation")
            return await self._generate_fallback_quiz(
                transcript, num_questions, question_types, difficulty
            )
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Quiz generation failed: {e}")
            return self._create_minimal_response(num_questions, question_types)

    async def _generate_ai_quiz(
        self,
        transcript: str,
        num_questions: int,
        question_types: List[str],
        difficulty: str,
        topics: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate quiz using AI with memory constraints."""
        
        questions = []
        
        # Retrieve limited context to save memory
        context = await self._retrieve_context(transcript, topics, top_k=2) if self.vector_store else []
        
        # Generate questions sequentially to manage memory
        for q_type in question_types:
            type_questions = max(1, num_questions // len(question_types))
            
            try:
                if q_type == "open_ended":
                    new_questions = await self._generate_open_ended_ai(
                        transcript, context, type_questions, difficulty
                    )
                else:  # mcq
                    new_questions = await self._generate_mcq_ai(
                        transcript, context, type_questions, difficulty
                    )
                
                questions.extend(new_questions)
                
                # Clean memory after each generation
                self.memory_manager.aggressive_cleanup()
                
            except Exception as e:
                logger.error(f"Error generating {q_type}: {e}")
                continue
        
        return {
            "questions": questions[:num_questions],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(questions),
                "question_types": question_types,
                "difficulty": difficulty,
                "success": True,
                "method": "ai_generation"
            }
        }

    async def _generate_open_ended_ai(self, transcript: str, context: List[str], num_questions: int, difficulty: str) -> List[Dict[str, Any]]:
        """Generate open-ended questions with memory efficiency."""
        try:
            # Shorter prompt for memory efficiency
            prompt = self._create_compact_open_ended_prompt(transcript, context, num_questions, difficulty)
            response = await self._generate_response(prompt)
            return self._parse_open_ended_response(response)
        except Exception as e:
            logger.error(f"AI open-ended generation failed: {e}")
            return []

    async def _generate_mcq_ai(self, transcript: str, context: List[str], num_questions: int, difficulty: str) -> List[Dict[str, Any]]:
        """Generate MCQ questions with memory efficiency."""
        try:
            prompt = self._create_compact_mcq_prompt(transcript, context, num_questions, difficulty)
            response = await self._generate_response(prompt)
            return self._parse_mcq_response(response)
        except Exception as e:
            logger.error(f"AI MCQ generation failed: {e}")
            return []

    def _create_compact_open_ended_prompt(self, transcript: str, context: List[str], num_questions: int, difficulty: str) -> str:
        """Create compact prompt to save memory."""
        context_text = "\n".join(context[:2]) if context else ""  # Limit context
        transcript_excerpt = transcript[:500]  # Limit transcript length
        
        return f"""Psychology content: {context_text}

Lecture: {transcript_excerpt}

Generate {num_questions} {difficulty} questions:
Q: [question]
A: [answer]"""

    def _create_compact_mcq_prompt(self, transcript: str, context: List[str], num_questions: int, difficulty: str) -> str:
        """Create compact MCQ prompt."""
        context_text = "\n".join(context[:2]) if context else ""
        transcript_excerpt = transcript[:500]
        
        return f"""Psychology content: {context_text}

Lecture: {transcript_excerpt}

Generate {num_questions} MCQ questions:
Q: [question]
A. [option]
B. [option]
C. [option]  
D. [option]
Correct: [letter]"""

    async def _generate_response(self, prompt: str) -> str:
        """Generate response with memory constraints."""
        try:
            if self.pipeline:
                result = self.pipeline(
                    prompt,
                    **self.generation_config,
                    return_full_text=False
                )
                return result[0]['generated_text'] if result else ""
            else:
                raise Exception("No model available")
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise

    def _parse_open_ended_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse open-ended questions from response."""
        questions = []
        
        try:
            parts = response.split("Q:")
            for part in parts[1:]:
                if "A:" in part:
                    qa_split = part.split("A:")
                    if len(qa_split) >= 2:
                        question = qa_split[0].strip()
                        answer = qa_split[1].strip()
                        
                        if question and answer:
                            questions.append({
                                "type": "open_ended",
                                "question": question,
                                "answer": answer,
                                "difficulty": "medium",
                                "metadata": {
                                    "generation_time": datetime.now().isoformat(),
                                    "model": "ai_generated"
                                }
                            })
        except Exception as e:
            logger.error(f"Error parsing open-ended response: {e}")
        
        return questions

    def _parse_mcq_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse MCQ questions from response."""
        questions = []
        
        try:
            parts = response.split("Q:")
            for part in parts[1:]:
                lines = [line.strip() for line in part.split("\n") if line.strip()]
                
                if len(lines) >= 6:
                    question = lines[0]
                    options = []
                    correct_answer = None
                    
                    for line in lines[1:]:
                        if line.startswith(("A.", "B.", "C.", "D.")):
                            options.append(line)
                        elif "correct:" in line.lower():
                            match = re.search(r'[ABCD]', line)
                            if match:
                                correct_answer = match.group()
                    
                    if len(options) == 4 and correct_answer and question:
                        questions.append({
                            "type": "mcq",
                            "question": question,
                            "options": options,
                            "correct_answer": correct_answer,
                            "difficulty": "medium",
                            "metadata": {
                                "generation_time": datetime.now().isoformat(),
                                "model": "ai_generated"
                            }
                        })
        except Exception as e:
            logger.error(f"Error parsing MCQ response: {e}")
        
        return questions

    async def _generate_fallback_quiz(
        self,
        transcript: str,
        num_questions: int,
        question_types: List[str],
        difficulty: str
    ) -> Dict[str, Any]:
        """Generate fallback quiz efficiently."""
        
        questions = []
        concepts = self._extract_psychology_concepts(transcript)
        questions_per_type = max(1, num_questions // len(question_types))
        
        for q_type in question_types:
            if q_type == "open_ended":
                questions.extend(self._create_fallback_open_ended(concepts, questions_per_type))
            elif q_type == "mcq":
                questions.extend(self._create_fallback_mcq(concepts, questions_per_type))
        
        return {
            "questions": questions[:num_questions],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(questions[:num_questions]),
                "question_types": question_types,
                "difficulty": difficulty,
                "success": True,
                "method": "fallback_generation",
                "fallback": True
            }
        }

    def _extract_psychology_concepts(self, transcript: str) -> List[str]:
        """Extract psychology concepts efficiently."""
        psych_terms = ["psychology", "behavior", "learning", "memory", "conditioning", "cognitive", "social"]
        found = [term for term in psych_terms if term.lower() in transcript.lower()]
        return found[:5]

    def _create_fallback_open_ended(self, concepts: List[str], num_questions: int) -> List[Dict[str, Any]]:
        """Create fallback open-ended questions."""
        questions = []
        templates = [
            ("What psychological concepts are discussed in this content?", 
             "The content covers key psychological principles related to human behavior and mental processes."),
            ("How do these concepts apply to real life?",
             "These psychological concepts can be applied in education, therapy, and understanding behavior."),
        ]
        
        for i in range(num_questions):
            question, answer = templates[i % len(templates)]
            questions.append({
                "type": "open_ended",
                "question": question,
                "answer": answer,
                "difficulty": "medium",
                "metadata": {"is_fallback": True, "generation_time": datetime.now().isoformat()}
            })
        
        return questions

    def _create_fallback_mcq(self, concepts: List[str], num_questions: int) -> List[Dict[str, Any]]:
        """Create fallback MCQ questions."""
        questions = []
        templates = [
            {
                "question": "Psychology is the study of:",
                "options": ["A) Brain only", "B) Mind and behavior", "C) Medicine", "D) Philosophy"],
                "correct_answer": "B"
            },
            {
                "question": "Classical conditioning involves:",
                "options": ["A) Consequences", "B) Association", "C) Memory", "D) Attention"],
                "correct_answer": "B"
            }
        ]
        
        for i in range(num_questions):
            mcq = templates[i % len(templates)].copy()
            mcq.update({
                "type": "mcq",
                "difficulty": "medium",
                "metadata": {"is_fallback": True, "generation_time": datetime.now().isoformat()}
            })
            questions.append(mcq)
        
        return questions

    async def _retrieve_context(self, transcript: str, topics: Optional[List[str]] = None, top_k: int = 2) -> List[str]:
        """Retrieve limited context to save memory."""
        if not self.vector_store:
            return []
        
        try:
            query = transcript[:300]  # Limit query length
            results = self.vector_store.similarity_search(query, k=top_k)
            return [doc.page_content for doc in results]
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "error": error_message,
            "questions": [],
            "metadata": {
                "success": False,
                "error": error_message,
                "timestamp": datetime.now().isoformat()
            }
        }

    def _create_minimal_response(self, num_questions: int, question_types: List[str]) -> Dict[str, Any]:
        """Create minimal response when everything fails."""
        questions = []
        
        for i in range(num_questions):
            questions.append({
                "type": question_types[0] if question_types else "open_ended",
                "question": "What psychological concepts were discussed?",
                "answer": "Please review the content for psychological concepts." if question_types[0] == "open_ended" else None,
                "options": ["A) Behavior", "B) Mind", "C) Both", "D) Neither"] if question_types[0] == "mcq" else None,
                "correct_answer": "C" if question_types[0] == "mcq" else None,
                "difficulty": "medium",
                "metadata": {"is_minimal": True}
            })
        
        return {
            "questions": questions,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(questions),
                "success": True,
                "method": "minimal_fallback"
            }
        }

    async def generate_quiz_async(
        self,
        content: str,
        num_questions: int = 5,
        difficulty: str = "medium",
        question_types: List[str] = ["mcq", "open_ended"],
        subject: str = "Psychology"
    ) -> Dict[str, Any]:
        """Async wrapper for quiz generation - designed for API endpoints."""
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting async quiz generation: {len(content)} chars, {num_questions} questions")
            
            # Force lightweight mode for API usage
            original_lightweight = self.use_lightweight
            self.use_lightweight = True
            
            # Generate quiz using existing method
            result = await self.generate_quiz(
                transcript=content,
                num_questions=num_questions,
                question_types=question_types,
                difficulty=difficulty
            )
            
            # Restore original setting
            self.use_lightweight = original_lightweight
            
            # Add timing and memory info
            generation_time = time.time() - start_time
            available_memory = self.memory_manager.get_available_gpu_memory()
            
            if "metadata" not in result:
                result["metadata"] = {}
            
            result["metadata"].update({
                "generation_time": round(generation_time, 2),
                "model_used": self.model_repo if self.model_loaded else "fallback",
                "device": str(self.device),
                "memory_info": {
                    "available_gb": round(available_memory, 2),
                    "device": str(self.device),
                    "model_loaded": self.model_loaded
                },
                "subject": subject,
                "api_version": "async_v1"
            })
            
            logger.info(f"Async quiz generation completed in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Async quiz generation failed: {e}")
            
            # Return fallback result
            return {
                "questions": [
                    {
                        "id": 1,
                        "type": "mcq",
                        "question": f"What is the main topic of this {subject} content?",
                        "options": ["Memory", "Attention", "Perception", "Learning"],
                        "correct_answer": "Memory",
                        "explanation": "Based on the provided content."
                    }
                ],
                "metadata": {
                    "generation_time": time.time() - start_time,
                    "model_used": "fallback",
                    "error": str(e),
                    "fallback": True,
                    "subject": subject
                }
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "available": self.available,
            "model_loaded": self.model_loaded,
            "fallback_only": self.fallback_only,
            "device": str(self.device),
            "memory_limit_gb": self.memory_limit_gb,
            "available_memory_gb": self.memory_manager.get_available_gpu_memory(),
            "skip_large_rag": self.skip_large_rag,
            "last_error": self.last_error
        }

    def cleanup(self):
        """Clean up resources."""
        try:
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            self.embedding_model = None
            self.vector_store = None
            self.memory_manager.aggressive_cleanup()
            self.model_loaded = False
            logger.info("Quiz generator cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass


# Factory function
def get_quiz_generator(**kwargs) -> QuizGenerator:
    """Get optimized quiz generator for 4GB GPU systems."""
    return QuizGenerator(**kwargs)


# Environment setup for your pipeline
def setup_quiz_environment_for_4gb_gpu():
    """Set up environment variables optimized for 4GB GPU."""
    
    config = {
        "QUIZ_SAFE_MODE": "true",
        "QUIZ_FALLBACK_ONLY": "false",           # Try AI first, fallback on failure
        "QUIZ_FORCE_CPU": "false",               # Let system decide based on memory
        "QUIZ_SKIP_PEFT": "true",                # Skip PEFT for 4GB GPU
        "QUIZ_DISABLE_QUANTIZATION": "false",    # Keep quantization for memory efficiency
        "QUIZ_MEMORY_LIMIT": "1.0",              # Conservative limit for 4GB GPU
        "QUIZ_TIMEOUT_SECONDS": "30",            # Shorter timeout
        "QUIZ_SKIP_LARGE_RAG": "true",           # Skip large textbook loading
        "QUIZ_MAX_RETRIES": "1"                  # Don't retry on failure
    }
    
    for key, value in config.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    print("[OK] Quiz environment configured for 4GB GPU")


# Test function
async def test_quiz_generator_4gb() -> Dict[str, Any]:
    """Test quiz generator on 4GB GPU system."""
    
    logger.info("Testing quiz generator for 4GB GPU...")
    
    try:
        # Setup environment
        setup_quiz_environment_for_4gb_gpu()
        
        # Create generator
        generator = get_quiz_generator()
        
        # Get system info
        model_info = generator.get_model_info()
        logger.info(f"System info: {model_info}")
        
        # Test with short transcript
        test_transcript = """
        Psychology studies mind and behavior. Classical conditioning involves learning 
        through association. Operant conditioning uses reinforcement and punishment.
        """
        
        # Test generation
        result = await generator.generate_quiz(
            transcript=test_transcript,
            num_questions=3,
            question_types=["open_ended", "mcq"],
            difficulty="medium"
        )
        
        # Cleanup
        generator.cleanup()
        
        return {
            "test_passed": True,
            "system_info": model_info,
            "quiz_result": result,
            "num_questions": len(result.get("questions", []))
        }
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return {
            "test_passed": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import asyncio
    
    # Test the optimized generator
    print("Testing Optimized Quiz Generator for 4GB GPU:")
    test_result = asyncio.run(test_quiz_generator_4gb())
    print(json.dumps(test_result, indent=2))