"""
Phase Manager for GPU Memory Optimization
Handles loading and unloading of AI models in phases
"""

import os
import torch
from typing import Dict, Optional, List, Any
import logging
from dataclasses import dataclass
from enum import Enum
from .gpu_memory_manager import ModelPriority, GPUMemoryManager

class SystemPhase(Enum):
    INITIALIZATION = "initialization"
    FACE_RECOGNITION = "face_recognition"
    SPEECH_PROCESSING = "speech_processing"
    ENGAGEMENT_ANALYSIS = "engagement_analysis"
    TRANSLATION = "translation"
    NOTE_GENERATION = "note_generation"
    TEXT_ALIGNMENT = "text_alignment"
    QUIZ_GENERATION = "quiz_generation"

@dataclass
class PhaseConfig:
    name: str
    models: List[str]
    memory_requirement: float  # GB
    priority: ModelPriority
    dependencies: List[str]
    is_loaded: bool = False

class PhaseManager:
    def __init__(self, gpu_manager: Optional[GPUMemoryManager] = None):
        self.current_phase: Optional[SystemPhase] = None
        self.gpu_manager = gpu_manager or GPUMemoryManager()
        self.phase_models: Dict[SystemPhase, Dict[str, Any]] = {
            SystemPhase.INITIALIZATION: {
                "models": [],
                "memory": 0.0,  # GB
                "priority": ModelPriority.CRITICAL
            },
            SystemPhase.FACE_RECOGNITION: {
                "models": ["face_detection", "face_recognition"],
                "memory": 1.5,  # GB
                "priority": ModelPriority.CRITICAL
            },
            SystemPhase.SPEECH_PROCESSING: {
                "models": ["asr_model", "punctuation_model"],
                "memory": 3.0,  # GB
                "priority": ModelPriority.HIGH
            },
            SystemPhase.ENGAGEMENT_ANALYSIS: {
                "models": ["student_monitor"],  # Use new integrated student monitoring
                "memory": 2.0,  # GB - higher memory for comprehensive analysis
                "priority": ModelPriority.HIGH
            },
            SystemPhase.TRANSLATION: {
                "models": ["translation_model"],
                "memory": 2.5,  # GB
                "priority": ModelPriority.MEDIUM
            },
            SystemPhase.NOTE_GENERATION: {
                "models": ["note_generation_model"],
                "memory": 1.0,  # GB
                "priority": ModelPriority.MEDIUM
            },
            SystemPhase.TEXT_ALIGNMENT: {
                "models": ["text_alignment_model"],
                "memory": 1.0,  # GB
                "priority": ModelPriority.LOW
            },
            SystemPhase.QUIZ_GENERATION: {
                "models": ["quiz_generator"],
                "memory": 1.0,  # GB
                "priority": ModelPriority.MEDIUM
            }
        }
        
        self.logger = logging.getLogger(__name__)
        self._register_models_with_memory_manager()
        self.current_phase = SystemPhase.INITIALIZATION

    def _register_models_with_memory_manager(self):
        """Register all models with the GPU memory manager."""
        try:
            for phase_name, phase_config in self.phase_models.items():
                for model_name in phase_config["models"]:
                    self.gpu_manager.register_model(
                        name=model_name,
                        memory_required=phase_config["memory"],
                        priority=phase_config["priority"]
                    )
            
            self.logger.info("Registered all models with GPU memory manager")
            
        except Exception as e:
            self.logger.error(f"Error registering models: {e}")
            raise

    def _can_load_phase(self, phase: SystemPhase) -> bool:
        """Check if a phase can be loaded based on dependencies and memory."""
        if phase == SystemPhase.INITIALIZATION:
            return True
            
        if phase not in self.phase_models:
            return False
            
        phase_config = self.phase_models[phase]
        
        # Check memory availability for all models in phase
        total_memory_required = phase_config["memory"]
        memory_info = self.gpu_manager.get_gpu_memory_info()
        
        if memory_info["free"] < total_memory_required:
            self.logger.warning(
                f"Insufficient memory for phase {phase.value}. "
                f"Required: {total_memory_required}GB, "
                f"Available: {memory_info['free']:.1f}GB"
            )
            return False
            
        return True

    async def switch_phase(self, phase: SystemPhase):
        """Switch to a different system phase, managing GPU resources"""
        if phase == self.current_phase:
            return
            
        try:
            # Special handling for initialization phase
            if phase == SystemPhase.INITIALIZATION:
                if self.current_phase:
                    await self._unload_phase(self.current_phase)
                self.current_phase = SystemPhase.INITIALIZATION
                return
                
            # Clear previous phase's resources
            if self.current_phase:
                await self._unload_phase(self.current_phase)
                
            # Check if we can load the new phase
            if not self._can_load_phase(phase):
                self.logger.error(f"Cannot load phase {phase.value} - insufficient resources")
                return
                
            # Load new phase's resources
            await self._load_phase(phase)
            self.current_phase = phase
            
            self.logger.info(f"Successfully switched to phase: {phase.value}")
            
        except Exception as e:
            self.logger.error(f"Error switching to phase {phase.value}: {e}")
            
    async def _load_phase(self, phase: SystemPhase):
        """Load models required for the specified phase"""
        with self.gpu_manager.optimize_memory():
            phase_config = self.phase_models[phase]
            
            # Pre-load cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            try:
                for model_name in phase_config["models"]:
                    if not self.gpu_manager.can_load_model(model_name):
                        self.logger.warning(f"Cannot load {model_name} - insufficient memory")
                        continue
                        
                    # Load model implementation here
                    # This is handled by the specific model classes
                    self.logger.info(f"Loading model: {model_name}")
                    
            except Exception as e:
                self.logger.error(f"Error loading models for phase {phase.value}: {e}")
                raise
                
    async def _unload_phase(self, phase: SystemPhase):
        """Unload models from the specified phase"""
        phase_config = self.phase_models[phase]
        
        try:
            for model_name in phase_config["models"]:
                # Unload model
                if self.gpu_manager.unload_model(model_name):
                    self.logger.info(f"Unloaded model: {model_name}")
                    
            self.gpu_manager.clear_memory()
            self.logger.info(f"Unloaded phase: {phase.value}")
            
        except Exception as e:
            self.logger.error(f"Error unloading phase {phase.value}: {e}")
            
    def get_current_phase(self) -> Optional[SystemPhase]:
        """Get the currently active system phase"""
        return self.current_phase
        
    def get_phase_status(self) -> Dict[str, Any]:
        """Get detailed status of current phase and memory usage"""
        status = {
            "current_phase": self.current_phase.value if self.current_phase else None,
            "gpu_memory": self.gpu_manager.get_gpu_memory_info(),
            "loaded_models": []
        }
        
        if self.current_phase:
            phase_config = self.phase_models[self.current_phase]
            status["loaded_models"] = [
                model for model in phase_config["models"]
                if self.gpu_manager.get_model(model) is not None
            ]
            
        return status
        
    async def cleanup(self):
        """Clean up all resources"""
        if self.current_phase:
            await self._unload_phase(self.current_phase)
        self.gpu_manager.cleanup_all()

# Global instance with default GPU memory manager
phase_manager = PhaseManager()
