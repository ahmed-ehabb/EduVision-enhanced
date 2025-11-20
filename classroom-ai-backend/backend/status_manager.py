"""
Manages the real-time status of various backend modules.

This file provides a centralized, lightweight way for different parts of the system
to report and check the availability of key features without creating circular
dependencies.

Attributes:
    DATABASE_AVAILABLE (bool): True if the database connection is live.
    ASR_AVAILABLE (bool): True if the Automatic Speech Recognition model is loaded.
    QUIZ_GENERATOR_AVAILABLE (bool): True if the Quiz Generator model is loaded.
    TRANSLATOR_AVAILABLE (bool): True if the Translation model is loaded.
    ENGAGEMENT_ANALYZER_AVAILABLE (bool): True if the Engagement Analyzer is ready.
    TEXT_ALIGNMENT_AVAILABLE (bool): True if the Text Alignment module is ready.
    NOTES_GENERATOR_AVAILABLE (bool): True if the Notes Generator model is loaded.
"""

# Database Status
DATABASE_AVAILABLE = False

# Model Status Flags
ASR_AVAILABLE = False
QUIZ_GENERATOR_AVAILABLE = False
TRANSLATOR_AVAILABLE = False
ENGAGEMENT_ANALYZER_AVAILABLE = False
TEXT_ALIGNMENT_AVAILABLE = False
NOTES_GENERATOR_AVAILABLE = False 