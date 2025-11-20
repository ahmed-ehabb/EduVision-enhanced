"""
Configuration parameters for the distraction detection system.
"""

# Path to the face dataset directory
FACE_DATASET = "facedataset"

# Time thresholds (in seconds)
NO_FACE_SECS = 5  # Time before alerting when no face is detected
OFF_SCREEN_SECS = 2  # Time before alerting when looking away

# Face recognition parameters
ID_TOLERANCE = 0.50  # Maximum distance threshold for face matching 