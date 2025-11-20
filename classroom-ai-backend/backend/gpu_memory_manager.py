"""
GPU Memory Manager for AI Models
Handles efficient loading and unloading of models to optimize GPU memory usage
"""

import os
import torch
import logging
import tensorflow as tf
import gc
import time
import psutil
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

class ModelPriority(Enum):
    """Priority levels for model loading."""
    CRITICAL = 1  # Must always be loaded (e.g., face recognition)
    HIGH = 2      # Load when needed, keep loaded (e.g., ASR)
    MEDIUM = 3    # Load when needed, can unload (e.g., translation)
    LOW = 4       # Load only when explicitly requested (e.g., quiz generation)

@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    memory_required: float  # GB
    priority: ModelPriority
    is_loaded: bool = False
    model: Optional[object] = None
    last_used: float = 0  # Timestamp
    peak_memory: float = 0  # Peak memory usage in GB
    batch_size: int = 1
    fp16_enabled: bool = False

class GPUMemoryManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, ModelInfo] = {}
        self.memory_threshold = 0.9  # 90% memory usage triggers cleanup
        self.tf_memory_growth = True
        self.memory_stats: List[Dict[str, float]] = []  # Track memory usage over time
        self.performance_metrics: Dict[str, List[float]] = {
            'load_times': [],
            'inference_times': [],
            'memory_efficiency': []
        }
        self._setup_tf()
        self._setup_torch()

    def _setup_tf(self):
        """Configure TensorFlow GPU memory growth"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, self.tf_memory_growth)
        except Exception as e:
            self.logger.warning(f"Warning: Could not configure TensorFlow GPU: {e}")

    def _setup_torch(self):
        """Configure PyTorch GPU memory management"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.warning(f"Warning: Could not configure PyTorch GPU: {e}")

    def setup_gpu_environment(self):
        """Configure environment variables for optimal GPU usage."""
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
        if self.is_gpu_available():
            self.logger.info("[ROCKET] GPU environment optimizations applied")
            self.print_gpu_info()
        else:
            self.logger.warning("[WARNING] No GPU detected, running in CPU mode")

    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return torch.cuda.is_available()

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get detailed GPU memory usage information."""
        if not self.is_gpu_available():
            return {"total": 0, "used": 0, "free": 0, "cached": 0}
            
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            total = props.total_memory / (1024**3)  # GB
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            cached = torch.cuda.memory_reserved(device) / (1024**3)
            
            # Get process-specific GPU memory
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / (1024**3)  # GB
            
            memory_info = {
                "total": total,
                "used": allocated,
                "free": total - allocated,
                "cached": cached,
                "reserved": reserved,
                "process": process_memory,
                "utilization": allocated / total * 100
            }
            
            # Record memory stats
            self.memory_stats.append({
                "timestamp": time.time(),
                **memory_info
            })
            
            return memory_info
        except Exception as e:
            self.logger.error(f"Error getting GPU memory info: {e}")
            return {"total": 0, "used": 0, "free": 0, "cached": 0}

    def print_gpu_info(self):
        """Print detailed GPU information."""
        if not self.is_gpu_available():
            return
            
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            memory = self.get_gpu_memory_info()
            
            self.logger.info("🖥️ GPU Information:")
            self.logger.info(f"Device: {props.name}")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"Total Memory: {memory['total']:.2f} GB")
            self.logger.info(f"Used Memory: {memory['used']:.2f} GB")
            self.logger.info(f"Free Memory: {memory['free']:.2f} GB")
            
        except Exception as e:
            self.logger.error(f"Error printing GPU info: {e}")

    def register_model(
        self,
        name: str,
        memory_required: float,
        priority: ModelPriority
    ) -> bool:
        """Register a model with the memory manager."""
        try:
            if name in self.models:
                self.logger.warning(f"Model {name} already registered")
                return False
                
            self.models[name] = ModelInfo(
                name=name,
                memory_required=memory_required,
                priority=priority
            )
            
            self.logger.info(f"Registered model {name} ({memory_required:.2f} GB required)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering model {name}: {e}")
            return False

    def can_load_model(self, name: str) -> bool:
        """Check if a model can be loaded into GPU memory."""
        if not self.is_gpu_available():
            return True  # Always return True for CPU mode
            
        try:
            if name not in self.models:
                return False
                
            model_info = self.models[name]
            if model_info.is_loaded:
                return True
                
            memory = self.get_gpu_memory_info()
            return memory["free"] >= model_info.memory_required
            
        except Exception as e:
            self.logger.error(f"Error checking if model {name} can be loaded: {e}")
            return False

    def load_model(self, name: str, model: object, enable_fp16: bool = True) -> bool:
        """Load a model into GPU memory with optimizations."""
        if name not in self.models:
            self.logger.error(f"Model {name} not registered")
            return False
            
        try:
            start_time = time.time()
            model_info = self.models[name]
            
            # Check memory and cleanup if needed
            if self.is_gpu_available():
                memory = self.get_gpu_memory_info()
                if memory["utilization"] > self.memory_threshold:
                    self.cleanup_memory(required_memory=model_info.memory_required)
            
            # Enable FP16 if supported
            if enable_fp16 and hasattr(model, 'half') and self.device.type == 'cuda':
                model = model.half()
                model_info.fp16_enabled = True
            
            # Move model to device
            if hasattr(model, 'to'):
                model.to(self.device)
            
            # Update model info
            model_info.model = model
            model_info.is_loaded = True
            model_info.last_used = time.time()
            
            # Record performance metrics
            load_time = time.time() - start_time
            self.performance_metrics['load_times'].append(load_time)
            
            # Update peak memory
            if self.is_gpu_available():
                current_memory = torch.cuda.memory_allocated() / (1024**3)
                model_info.peak_memory = max(model_info.peak_memory, current_memory)
            
            self.logger.info(
                f"Loaded model {name} to {self.device} "
                f"(FP16: {model_info.fp16_enabled}, "
                f"Load Time: {load_time:.2f}s)"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model {name}: {e}")
            return False

    def unload_model(self, name: str) -> bool:
        """Unload a model from GPU memory."""
        if name not in self.models:
            return False
            
        try:
            model_info = self.models[name]
            
            # Skip if model is CRITICAL priority
            if model_info.priority == ModelPriority.CRITICAL:
                return False
            
            # Move model to CPU if possible
            if model_info.model is not None and hasattr(model_info.model, 'to'):
                model_info.model.to('cpu')
            
            model_info.model = None
            model_info.is_loaded = False
            
            if self.is_gpu_available():
                torch.cuda.empty_cache()
            
            self.logger.info(f"Unloaded model {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unloading model {name}: {e}")
            return False

    def cleanup_memory(self, required_memory: Optional[float] = None):
        """Enhanced memory cleanup with monitoring."""
        if not self.is_gpu_available():
            return
            
        try:
            initial_memory = self.get_gpu_memory_info()
            self.logger.info(f"Starting cleanup. Initial memory usage: {initial_memory['used']:.2f}GB")
            
            # Sort models by priority and last used time
            models_to_unload = sorted(
                [m for m in self.models.values() if m.is_loaded],
                key=lambda x: (x.priority.value, -x.last_used)
            )
            
            freed_memory = 0
            unloaded_models = []
            
            for model in models_to_unload:
                if model.priority == ModelPriority.CRITICAL:
                    continue
                
                if required_memory is not None:
                    if initial_memory["free"] + freed_memory >= required_memory:
                        break
                
                if self.unload_model(model.name):
                    freed_memory += model.memory_required
                    unloaded_models.append(model.name)
                
                current_usage = initial_memory["used"] - freed_memory
                if current_usage / initial_memory["total"] < self.memory_threshold:
                    break
            
            # Force garbage collection and cache clearing
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            final_memory = self.get_gpu_memory_info()
            actual_freed = initial_memory["used"] - final_memory["used"]
            
            self.logger.info(
                f"Cleanup completed:\n"
                f"- Freed memory: {actual_freed:.2f}GB\n"
                f"- Unloaded models: {', '.join(unloaded_models)}\n"
                f"- Current usage: {final_memory['used']:.2f}GB ({final_memory['utilization']:.1f}%)"
            )
            
            # Record cleanup effectiveness
            self.performance_metrics['memory_efficiency'].append(
                actual_freed / initial_memory["used"] if initial_memory["used"] > 0 else 0
            )
            
        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")

    def get_model(self, name: str) -> Optional[object]:
        """Get a loaded model."""
        if name not in self.models:
            return None
            
        model_info = self.models[name]
        if not model_info.is_loaded:
            return None
            
        # Update last used time
        model_info.last_used = time.time()
        return model_info.model

    def cleanup_all(self):
        """Clean up all models and free GPU memory."""
        try:
            for name in list(self.models.keys()):
                self.unload_model(name)
            
            if self.is_gpu_available():
                torch.cuda.empty_cache()
                
            self.logger.info("Cleaned up all models")
            
        except Exception as e:
            self.logger.error(f"Error during final cleanup: {e}")

    @contextmanager
    def optimize_memory(self):
        """Context manager for optimized GPU memory usage"""
        try:
            # Clear memory before loading models
            self.clear_memory()
            yield
        finally:
            # Cleanup after model operations
            self.clear_memory()
            
    def clear_memory(self):
        """Clear GPU memory"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear TensorFlow memory
            if tf.config.experimental.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
                
            # Force garbage collection
            gc.collect()
        except Exception as e:
            self.logger.warning(f"Warning: Error clearing GPU memory: {e}")
            
    def cleanup(self):
        """Alias for cleanup_all for backward compatibility."""
        self.cleanup_all()

    def get_memory_requirement(self, model_name: str) -> float:
        """Get the memory requirement for a model in GB."""
        # Default memory requirements for different models
        memory_requirements = {
            'face_recognition': 1.5,  # Face recognition model
            'engagement_classifier': 2.0,  # Engagement classification model
            'asr_model': 3.0,  # Speech recognition model
            'translation_model': 2.5,  # Translation model
            'quiz_generator': 4.0  # Quiz generation model
        }
        
        return memory_requirements.get(model_name, 1.0)  # Default to 1GB if unknown

    def get_priority(self, model_name: str) -> ModelPriority:
        """Get the priority level for a model."""
        # Define priority levels for different models
        priority_levels = {
            'face_recognition': ModelPriority.CRITICAL,  # Always needed
            'engagement_classifier': ModelPriority.HIGH,  # Frequently used
            'asr_model': ModelPriority.HIGH,  # Real-time processing
            'translation_model': ModelPriority.MEDIUM,  # Can be loaded on demand
            'quiz_generator': ModelPriority.LOW  # Used occasionally
        }
        
        return priority_levels.get(model_name, ModelPriority.MEDIUM)  # Default to MEDIUM priority

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        metrics = {
            'models': {},
            'system': {
                'avg_load_time': sum(self.performance_metrics['load_times']) / len(self.performance_metrics['load_times'])
                if self.performance_metrics['load_times'] else 0,
                'cleanup_efficiency': sum(self.performance_metrics['memory_efficiency']) / len(self.performance_metrics['memory_efficiency'])
                if self.performance_metrics['memory_efficiency'] else 0,
                'memory_usage_history': self.memory_stats[-10:] if self.memory_stats else []
            }
        }
        
        # Per-model metrics
        for name, model_info in self.models.items():
            metrics['models'][name] = {
                'priority': model_info.priority.value,
                'is_loaded': model_info.is_loaded,
                'peak_memory_gb': model_info.peak_memory,
                'fp16_enabled': model_info.fp16_enabled,
                'batch_size': model_info.batch_size,
                'last_used': model_info.last_used
            }
        
        return metrics

# Global instance
gpu_memory_manager = GPUMemoryManager()
