#!/usr/bin/env python
"""
GPU Configuration and Optimization for Classroom AI Backend
"""
import os
import torch
import tensorflow as tf
import logging
from typing import Optional, Dict, Any
import GPUtil
from contextlib import contextmanager
import gc
import numpy as np


def setup_gpu_environment():
    """Configure environment variables for optimal GPU usage."""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"  # Optimize TF thread handling
    os.environ["TF_GPU_THREAD_COUNT"] = "4"  # Limit TF threads
    os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"  # Enable persistent algorithms
    print("[ROCKET] GPU environment optimizations applied")


def get_device_info():
    """Get detailed device information."""
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            return {
                "device": "cuda",
                "device_count": torch.cuda.device_count(),
                "current_device": device,
                "device_name": props.name,
                "memory_total_gb": round(props.total_memory / 1024**3, 2),
                "memory_allocated_gb": round(
                    torch.cuda.memory_allocated(device) / 1024**3, 2
                ),
                "memory_cached_gb": round(
                    torch.cuda.memory_reserved(device) / 1024**3, 2
                ),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        else:
            return {"device": "cpu", "reason": "CUDA not available"}
    except ImportError:
        return {"device": "cpu", "reason": "PyTorch not installed"}


def print_device_status():
    """Print detailed device status."""
    info = get_device_info()
    print("🖥️  Device Status:")
    print("=" * 50)
    if info["device"] == "cuda":
        print(f"[OK] GPU: {info['device_name']}")
        print(f"🔢 Device Count: {info['device_count']}")
        print(f"[DART] Current Device: {info['current_device']}")
        print(f"💾 Total Memory: {info['memory_total_gb']} GB")
        print(f"[CHART] Allocated Memory: {info['memory_allocated_gb']} GB")
        print(f"🗂️  Cached Memory: {info['memory_cached_gb']} GB")
        print(f"[TOOL] CUDA Version: {info['cuda_version']}")
        print(f"🧠 cuDNN Version: {info['cudnn_version']}")
    else:
        print(f"[WARNING]  Using CPU: {info.get('reason', 'Unknown reason')}")
    print("=" * 50)


def optimize_model_loading():
    """Apply optimizations for model loading."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)
            print("🧹 GPU cache cleared and memory optimized")
            return True
    except ImportError:
        pass
    return False


class GPUConfig:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = None
        self.memory_fraction = float(os.getenv('GPU_MEMORY_FRACTION', '0.8'))
        self.setup_gpu()
        
    def setup_gpu(self):
        """Configure GPU settings for all frameworks"""
        try:
            # Configure PyTorch first
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                # Enable TF32 for better performance on Ampere GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cuda.allow_tf32 = True
                # Enable autocast for FP16 - Updated to use new API
                torch.amp.autocast('cuda', enabled=True)
                self.logger.info(f"PyTorch using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                self.logger.warning("PyTorch using CPU - No GPU available")
            
            # Configure TensorFlow
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Calculate memory limit (30% of total memory for TensorFlow)
                    gpu_memory = self._get_gpu_memory()  # in MB
                    memory_limit = int(0.3 * gpu_memory)  # 30% of total memory
                    
                    # Configure virtual devices for better memory management
                    tf.config.set_logical_device_configuration(
                        gpus[0],
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
                    
                    # Enable mixed precision globally
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    
                    # Enable XLA optimization
                    tf.config.optimizer.set_jit(True)
                    
                    # Configure TF memory allocator
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    tf.config.experimental.enable_tensor_float_32_execution(True)
                    
                    self.logger.info(f"TensorFlow configured for GPU with {memory_limit}MB limit and mixed precision")
                except RuntimeError as e:
                    self.logger.error(f"TensorFlow GPU configuration error: {e}")
            else:
                self.logger.warning("TensorFlow using CPU - No GPU available")
                
        except Exception as e:
            self.logger.error(f"Error setting up GPU: {e}")
            self.device = torch.device('cpu')
            
    def _get_gpu_memory(self) -> int:
        """Get available GPU memory in MB"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryTotal
            return 0
        except Exception:
            return 0
            
    @contextmanager
    def device_context(self, model_name: str):
        """Context manager for GPU memory management"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if torch.cuda.memory_allocated() > 0.9 * torch.cuda.get_device_properties(0).total_memory:
                    self.logger.warning(f"High GPU memory usage before loading {model_name}")
                    gc.collect()
                    torch.cuda.empty_cache()
            yield self.device
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get GPU configuration for specific model types"""
        config = {
            'device': self.device,
            'use_cuda': self.device.type == 'cuda',
            'memory_fraction': self.memory_fraction,
            'use_mixed_precision': True,  # Enable by default for all models
            'optimize_memory': True,
            'batch_processing': True  # Enable batch processing by default
        }
        
        if model_type == 'face_recognition':
            config.update({
                'batch_size': 32 if self.device.type == 'cuda' else 8,
                'num_workers': 4 if self.device.type == 'cuda' else 0,
                'memory_limit': '1.5GB',
                'fp16_inference': True,
                'optimize_transforms': True
            })
        elif model_type == 'engagement':
            config.update({
                'batch_size': 16 if self.device.type == 'cuda' else 4,
                'use_fp16': self.device.type == 'cuda',
                'memory_limit': '1.2GB',
                'enable_dynamic_shapes': True
            })
        elif model_type == 'translation':
            config.update({
                'batch_size': 24 if self.device.type == 'cuda' else 6,
                'use_cache': True,
                'memory_limit': '2.5GB',
                'fp16_inference': True,
                'optimize_decoder': True
            })
        elif model_type == 'quiz_generation':
            config.update({
                'batch_size': 4 if self.device.type == 'cuda' else 1,  # Increased for better throughput
                'use_4bit': True,  # Enable 4-bit quantization
                'use_nested_quant': True,  # Enable nested quantization
                'bnb_4bit_compute_dtype': 'float16',
                'memory_limit': '1.0GB',  # Reserve 1GB for quiz generation
                'enable_attention_slicing': True,
                'use_bettertransformer': True
            })
            
        return config
        
    def get_device_info(self) -> Dict[str, Any]:
        """Get current GPU/device information"""
        info = {
            'device_type': self.device.type,
            'memory_fraction': self.memory_fraction,
            'pytorch_gpu': torch.cuda.is_available(),
            'tensorflow_gpu': len(tf.config.list_physical_devices('GPU')) > 0
        }
        
        if self.device.type == 'cuda':
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory': self._get_gpu_memory(),
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version()
            })
            
        return info
        
    def optimize_model_placement(self, models: Dict[str, Any]):
        """Optimize multiple models across available GPU memory"""
        if not torch.cuda.is_available():
            return
            
        total_memory = self._get_gpu_memory()
        available_memory = total_memory * self.memory_fraction
        
        # Sort models by priority/size
        sorted_models = sorted(
            models.items(),
            key=lambda x: getattr(x[1], 'gpu_priority', 0),
            reverse=True
        )
        
        # Allocate memory based on priority
        for name, model in sorted_models:
            try:
                if hasattr(model, 'to'):
                    model.to(self.device)
                self.logger.info(f"Moved {name} to {self.device}")
            except Exception as e:
                self.logger.error(f"Error moving {name} to GPU: {e}")
                if hasattr(model, 'to'):
                    model.to('cpu')
                    
    def cleanup(self):
        """Clean up GPU resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
# Global instance
gpu_config = GPUConfig()


if __name__ == "__main__":
    setup_gpu_environment()
    print_device_status()
    optimize_model_loading()
