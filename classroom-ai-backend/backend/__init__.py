"""
Classroom AI Backend Initialization
"""

import logging
import os
import sys

# Add backend to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Optional: Disable verbose logging from libraries
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

__version__ = "1.3.0"
__author__ = "Ahmed"

# Core module availability flags
__all__ = [
    "api_server",
    "asr_module",
    "notes_generator",
    "translation_module",
    "text_alignment",
    "engagement_analyzer",
    "quiz_generator",
    "gpu_memory_manager",
    "gpu_config",
    "language_stats",
]
