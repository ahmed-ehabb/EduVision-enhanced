"""
AI Model Manager for Efficient GPU Utilization
Handles on-demand loading, unloading, and phase management of all AI models
"""

import gc
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import psutil

# Import GPU configuration
from backend.gpu_config import get_device_info, optimize_model_loading

# Import all AI modules
from backend.asr_module import ASRModule
from backend.notes_generator import LectureNotesGenerator
from backend.quiz_generator import QuizGenerator
from backend.text_alignment import TextAlignmentAnalyzer
from backend.translation_module import TranslationModel
from backend.face_recognition_system import FaceRecognitionSystem
from backend.student_monitor_integration import StudentMonitorIntegration
from backend.engagement_analyzer import EngagementAnalyzer


class ModelPhase(Enum):
    """Different phases of model execution"""
    AUDIO_PROCESSING = "audio_processing"
    TEXT_ANALYSIS = "text_analysis"
    FACE_RECOGNITION = "face_recognition"
    ENGAGEMENT_ANALYSIS = "engagement_analysis"
    QUIZ_GENERATION = "quiz_generation"
    IDLE = "idle"


class ModelPriority(Enum):
    """Model loading priority levels"""
    CRITICAL = 1    # Must always be loaded (e.g., ASR for live sessions)
    HIGH = 2        # Load when needed, keep in memory
    MEDIUM = 3      # Load on demand, may unload
    LOW = 4         # Load only when explicitly requested


class AIModelManager:
    """
    Centralized manager for all AI models with GPU-efficient loading and phase management
    """
    
    def __init__(self, max_gpu_memory_usage: float = 0.85, cache_timeout: int = 300):
        self.logger = logging.getLogger(__name__)
        self.max_gpu_memory_usage = max_gpu_memory_usage
        self.cache_timeout = cache_timeout  # seconds
        
        # Model registry
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.model_last_used: Dict[str, datetime] = {}
        self.model_loading_lock = threading.Lock()
        
        # Phase management
        self.current_phase = ModelPhase.IDLE
        self.phase_models: Dict[ModelPhase, List[str]] = {}
        
        # Memory tracking
        self.gpu_info = get_device_info()
        self.memory_usage = {}
        
        # Initialize model configurations
        self._initialize_model_configs()
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def _initialize_model_configs(self):
        """Initialize configuration for all models"""
        self.model_configs = {
            "asr": {
                "class": ASRModule,
                "priority": ModelPriority.CRITICAL,
                "phases": [ModelPhase.AUDIO_PROCESSING],
                "gpu_memory_mb": 1500,
                "init_args": {},
                "warmup": True
            },
            "translation": {
                "class": TranslationModel,
                "priority": ModelPriority.HIGH,
                "phases": [ModelPhase.AUDIO_PROCESSING, ModelPhase.TEXT_ANALYSIS],
                "gpu_memory_mb": 1000,
                "init_args": {},
                "warmup": True
            },
            "notes_generator": {
                "class": LectureNotesGenerator,
                "priority": ModelPriority.HIGH,
                "phases": [ModelPhase.TEXT_ANALYSIS],
                "gpu_memory_mb": 800,
                "init_args": {},
                "warmup": False
            },
            "quiz_generator": {
                "class": QuizGenerator,
                "priority": ModelPriority.MEDIUM,
                "phases": [ModelPhase.QUIZ_GENERATION],
                "gpu_memory_mb": 800,
                "init_args": {},
                "warmup": False
            },
            "text_alignment": {
                "class": TextAlignmentAnalyzer,
                "priority": ModelPriority.MEDIUM,
                "phases": [ModelPhase.TEXT_ANALYSIS],
                "gpu_memory_mb": 500,
                "init_args": {},
                "warmup": False
            },
            "face_recognition": {
                "class": FaceRecognitionSystem,
                "priority": ModelPriority.HIGH,
                "phases": [ModelPhase.FACE_RECOGNITION],
                "gpu_memory_mb": 600,
                "init_args": {},
                "warmup": True
            },
            "student_monitor": {
                "class": StudentMonitorIntegration,
                "priority": ModelPriority.HIGH,
                "phases": [ModelPhase.ENGAGEMENT_ANALYSIS],
                "gpu_memory_mb": 2000,  # Higher memory for comprehensive analysis
                "init_args": {"gpu_manager": None},  # Will be set during initialization
                "warmup": True
            },
            "engagement_analyzer": {
                "class": EngagementAnalyzer,
                "priority": ModelPriority.LOW,  # Lower priority since we have student_monitor
                "phases": [ModelPhase.ENGAGEMENT_ANALYSIS],
                "gpu_memory_mb": 300,
                "init_args": {},
                "warmup": False
            }
        }
        
        # Initialize phase to model mapping
        for model_name, config in self.model_configs.items():
            for phase in config["phases"]:
                if phase not in self.phase_models:
                    self.phase_models[phase] = []
                self.phase_models[phase].append(model_name)
    
    async def load_model(self, model_name: str, force_reload: bool = False) -> bool:
        """Load a specific model"""
        try:
            with self.model_loading_lock:
                # Check if model is already loaded
                if model_name in self.models and not force_reload:
                    self.model_last_used[model_name] = datetime.now()
                    return True
                
                # Check if model config exists
                if model_name not in self.model_configs:
                    self.logger.error(f"Model '{model_name}' not found in configurations")
                    return False
                
                config = self.model_configs[model_name]
                
                # Check GPU memory before loading
                if not self._check_gpu_memory(config["gpu_memory_mb"]):
                    self.logger.warning(f"Insufficient GPU memory for {model_name}, attempting cleanup")
                    await self._free_memory_for_model(config["gpu_memory_mb"])
                
                # Load the model
                self.logger.info(f"🔄 Loading model: {model_name}")
                
                try:
                    model_class = config["class"]
                    init_args = config["init_args"].copy()
                    
                    # Special handling for student_monitor to pass GPU manager
                    if model_name == "student_monitor":
                        # Initialize GPU manager if not present
                        if not hasattr(self, 'gpu_memory_manager'):
                            from .gpu_memory_manager import GPUMemoryManager
                            self.gpu_memory_manager = GPUMemoryManager()
                        init_args["gpu_manager"] = self.gpu_memory_manager
                    
                    model_instance = model_class(**init_args)
                    
                    # Warmup if needed
                    if config.get("warmup", False):
                        await self._warmup_model(model_name, model_instance)
                    
                    self.models[model_name] = model_instance
                    self.model_last_used[model_name] = datetime.now()
                    
                    self.logger.info(f"[OK] Successfully loaded model: {model_name}")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"[ERROR] Error loading model {model_name}: {e}")
                    return False
        
        except Exception as e:
            self.logger.error(f"Error in load_model for {model_name}: {e}")
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a specific model"""
        try:
            with self.model_loading_lock:
                if model_name not in self.models:
                    return True
                
                # Don't unload critical models
                config = self.model_configs.get(model_name, {})
                if config.get("priority") == ModelPriority.CRITICAL:
                    self.logger.info(f"Skipping unload of critical model: {model_name}")
                    return False
                
                # Cleanup model
                model = self.models[model_name]
                if hasattr(model, 'cleanup'):
                    model.cleanup()
                
                del self.models[model_name]
                if model_name in self.model_last_used:
                    del self.model_last_used[model_name]
                
                # Force garbage collection
                gc.collect()
                
                # Clear GPU cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                
                self.logger.info(f"[TRASH] Unloaded model: {model_name}")
                return True
        
        except Exception as e:
            self.logger.error(f"Error unloading model {model_name}: {e}")
            return False
    
    async def set_phase(self, phase: ModelPhase) -> bool:
        """Set the current processing phase and load required models"""
        try:
            if self.current_phase == phase:
                return True
            
            self.logger.info(f"🔄 Switching to phase: {phase.value}")
            
            # Get models needed for this phase
            required_models = self.phase_models.get(phase, [])
            
            # Load required models
            load_tasks = []
            for model_name in required_models:
                if model_name not in self.models:
                    load_tasks.append(self.load_model(model_name))
            
            if load_tasks:
                results = await asyncio.gather(*load_tasks, return_exceptions=True)
                failed_loads = [i for i, result in enumerate(results) if not result or isinstance(result, Exception)]
                
                if failed_loads:
                    self.logger.warning(f"Failed to load some models for phase {phase.value}")
            
            # Unload models not needed for any active phase
            await self._cleanup_unused_models()
            
            self.current_phase = phase
            self.logger.info(f"[OK] Phase switched to: {phase.value}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error setting phase {phase}: {e}")
            return False
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a model instance (load if necessary)"""
        try:
            if model_name not in self.models:
                # Try to load the model synchronously
                asyncio.create_task(self.load_model(model_name))
                return None
            
            self.model_last_used[model_name] = datetime.now()
            return self.models[model_name]
        
        except Exception as e:
            self.logger.error(f"Error getting model {model_name}: {e}")
            return None
    
    @contextmanager
    def use_model(self, model_name: str):
        """Context manager for using a model"""
        model = self.get_model(model_name)
        try:
            yield model
        finally:
            # Update last used time
            if model_name in self.models:
                self.model_last_used[model_name] = datetime.now()
    
    async def _warmup_model(self, model_name: str, model_instance: Any):
        """Warmup a model with dummy data"""
        try:
            self.logger.info(f"🔥 Warming up model: {model_name}")
            
            if model_name == "asr":
                # Warmup ASR with silence
                import numpy as np
                dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
                if hasattr(model_instance, 'transcribe_audio'):
                    await asyncio.to_thread(model_instance.transcribe_audio, dummy_audio)
            
            elif model_name == "face_recognition":
                # Warmup with dummy image
                import numpy as np
                dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
                if hasattr(model_instance, 'recognize_faces'):
                    await asyncio.to_thread(model_instance.recognize_faces, dummy_image)
            
            elif model_name == "student_monitor":
                # Warmup student monitoring with dummy frame
                import numpy as np
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                if hasattr(model_instance, 'analyze_student_frame'):
                    await model_instance.analyze_student_frame(dummy_frame, "warmup_student")
            
            # Add more warmup routines as needed
            
        except Exception as e:
            self.logger.warning(f"Warmup failed for {model_name}: {e}")
    
    def _check_gpu_memory(self, required_mb: int) -> bool:
        """Check if enough GPU memory is available"""
        try:
            if self.gpu_info.get("device") != "cuda":
                return True  # No GPU memory limits on CPU
            
            import torch
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
                available_memory = total_memory - current_memory
                
                return available_memory >= required_mb
            
            return True
        
        except Exception as e:
            self.logger.warning(f"Could not check GPU memory: {e}")
            return True
    
    async def _free_memory_for_model(self, required_mb: int):
        """Free memory by unloading low-priority models"""
        try:
            # Sort models by priority and last used time
            unload_candidates = []
            
            for model_name, last_used in self.model_last_used.items():
                config = self.model_configs.get(model_name, {})
                priority = config.get("priority", ModelPriority.MEDIUM)
                
                if priority != ModelPriority.CRITICAL:
                    age = (datetime.now() - last_used).total_seconds()
                    unload_candidates.append((model_name, priority.value, age))
            
            # Sort by priority (higher value = lower priority) and age
            unload_candidates.sort(key=lambda x: (x[1], -x[2]))
            
            # Unload models until we have enough memory
            for model_name, _, _ in unload_candidates:
                if self._check_gpu_memory(required_mb):
                    break
                await self.unload_model(model_name)
        
        except Exception as e:
            self.logger.error(f"Error freeing memory: {e}")
    
    async def _cleanup_unused_models(self):
        """Cleanup models that haven't been used recently"""
        try:
            current_time = datetime.now()
            timeout_delta = timedelta(seconds=self.cache_timeout)
            
            models_to_unload = []
            for model_name, last_used in self.model_last_used.items():
                if current_time - last_used > timeout_delta:
                    config = self.model_configs.get(model_name, {})
                    if config.get("priority") != ModelPriority.CRITICAL:
                        models_to_unload.append(model_name)
            
            for model_name in models_to_unload:
                await self.unload_model(model_name)
        
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # Check every minute
                    await self._cleanup_unused_models()
                except Exception as e:
                    self.logger.error(f"Error in cleanup loop: {e}")
        
        asyncio.create_task(cleanup_loop())
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                "current_phase": self.current_phase.value,
                "loaded_models": list(self.models.keys()),
                "gpu_info": self.gpu_info,
                "memory_usage": {},
                "model_status": {}
            }
            
            # Get memory usage if possible
            try:
                import torch
                if torch.cuda.is_available():
                    status["memory_usage"] = {
                        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                        "cached_mb": torch.cuda.memory_reserved() / 1024**2,
                        "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2
                    }
            except ImportError:
                pass
            
            # Model status
            for model_name, config in self.model_configs.items():
                status["model_status"][model_name] = {
                    "loaded": model_name in self.models,
                    "priority": config["priority"].name,
                    "phases": [phase.value for phase in config["phases"]],
                    "last_used": self.model_last_used.get(model_name),
                    "gpu_memory_mb": config["gpu_memory_mb"]
                }
            
            return status
        
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown the model manager and cleanup all models"""
        try:
            self.logger.info("🔄 Shutting down AI Model Manager...")
            
            # Unload all models
            model_names = list(self.models.keys())
            for model_name in model_names:
                await self.unload_model(model_name)
            
            # Clear caches
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            self.logger.info("[OK] AI Model Manager shutdown complete")
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Global instance
ai_model_manager = AIModelManager() 