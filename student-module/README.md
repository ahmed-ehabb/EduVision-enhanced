# Distraction Detection System

A real-time computer vision system for monitoring student engagement and detecting distractions during learning sessions.

## Features

- Real-time face detection and tracking using MediaPipe Face Mesh
- Identity verification system with periodic checks
- Multi-factor distraction detection:
  - Head pose estimation
  - Eye tracking and blink detection
  - Yawn detection
  - Posture analysis
- Engagement level monitoring
- Comprehensive alert system
- Session statistics and logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/distraction-detection-system.git
cd distraction-detection-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python run_local_cv.py
```

2. Follow the on-screen instructions for identity verification
3. The system will automatically monitor for distractions and engagement levels

## System Requirements

- Python 3.7+
- Webcam (minimum 720p)
- Good lighting conditions
- Windows OS (for audio alerts)

## Project Structure

```
distractionModel-master/
├── run_local_cv.py           # Main application file
├── util/
│   ├── analysis_realtime_cv.py  # Face detection and analysis
│   ├── identity_verifier.py     # Identity verification
│   ├── gaze_monitor.py          # Gaze tracking (deprecated)
│   ├── afk_timer.py            # AFK detection (deprecated)
│   └── config.py               # Configuration parameters
├── logs/                      # Session logs directory
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── documentation.md           # Detailed documentation
```

## Configuration

Key parameters can be adjusted in `run_local_cv.py`:

```python
WARNING_INTERVAL = 45          # Seconds before showing "Please refocus"
DOCTOR_ALERT_INTERVAL = 120    # Seconds before alerting doctor
NO_FACE_ALERT_INTERVAL = 30    # Seconds before alerting for no face
ID_VERIFICATION_INTERVAL = 600 # 10 minutes between identity checks
DISTRACTION_THRESHOLD = 5.0    # Seconds before considering someone distracted
```

## Documentation

For detailed documentation, including:
- Technical architecture
- Algorithms and techniques
- Performance metrics
- Configuration parameters
- Input/Output specifications

Please refer to [documentation.md](documentation.md)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MediaPipe for face mesh detection
- OpenCV for computer vision capabilities
- All contributors and users of the system

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/distraction-detection-system](https://github.com/yourusername/distraction-detection-system)
