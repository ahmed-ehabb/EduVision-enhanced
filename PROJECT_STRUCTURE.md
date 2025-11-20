# EduVision - Project Structure

## Overview
This document provides a complete overview of the project structure and what each component contains.

## Root Directory

```
EduVision-enhanced/
├── .git/                   # Git repository
├── .gitignore             # Comprehensive ignore rules
├── README.md              # Main project documentation
├── LICENSE                # MIT License
├── CONTRIBUTING.md        # Contribution guidelines
├── START_SERVERS.md       # Server startup guide
├── QUICK_FIX.md           # PostgreSQL troubleshooting
├── api/                   # Unified API (Port 8002)
├── classroom-ai-backend/  # Teacher Module (Port 8001)
├── student-module/        # Student Module (Port 8001)
├── frontend/              # React Application (Port 3000)
└── database/              # Database schema & setup
```

## Module Breakdown

### 1. Unified API (`api/`)
**Purpose**: Authentication, user management, and data persistence

**Key Files**:
- `main.py` - FastAPI application with 17 endpoints
- `models.py` - 19 SQLAlchemy ORM models
- `crud.py` - Database operations
- `auth.py` - JWT + Bcrypt authentication
- `database.py` - PostgreSQL connection
- `setup_db.py` - Database initialization script
- `init_db.py` - Create test users
- `requirements.txt` - Python dependencies

**Endpoints**:
- Authentication: `/auth/login`, `/auth/register`, `/auth/refresh`
- Lectures: `/api/lectures`
- Sessions: `/api/sessions`
- Analytics: `/api/sessions/{id}/analytics`

### 2. Teacher Module (`classroom-ai-backend/`)
**Purpose**: AI-powered lecture processing

**Key Files**:
- `main.py` - FastAPI application with 9 endpoints
- `backend/teacher_module_v2.py` - AI pipeline orchestration
- `backend/asr_module.py` - Whisper speech recognition
- `backend/translation_module.py` - NLLB translation
- `backend/notes_generator_v2.py` - Notes generation
- `backend/quiz_generator_v2.py` - Llama 3.2 quiz generation
- `backend/alignment_module.py` - Content alignment (SBERT)
- `requirements.txt` - Python dependencies
- `docker-compose.yml` - Docker configuration
- `Dockerfile` - Container setup

**AI Models**:
- Whisper Large V3 (4-bit quantized)
- NLLB-200 (translation)
- Llama 3.2 3B (quiz/notes generation)
- SBERT (content alignment)

**Processing Steps**:
1. Transcription (ASR)
2. Translation
3. Engagement Analysis
4. Content Alignment
5. Notes Generation
6. Quiz Generation
7. Report Creation

### 3. Student Module (`student-module/`)
**Purpose**: Real-time engagement and distraction monitoring

**Key Files**:
- `fastapi_ui.py` - FastAPI application (main server)
- `student_module.py` - Core distraction detection
- `emotion_recognition.py` - Emotion analysis
- `Models/` - ML models directory
  - Haar Cascade (face detection)
  - LBPH (face recognition)
  - CNN (emotion classification)
- `templates/` - HTML templates
- `static/` - CSS/JS assets
- `requirements.txt` - Python dependencies

**Detection Capabilities**:
- Face presence/absence
- Face identity (recognition)
- Emotions (7 types)
- Head pose/orientation
- Eye gaze direction
- Yawning detection
- Blink rate monitoring
- Hand-to-face gestures

### 4. Frontend (`frontend/`)
**Purpose**: React-based web application

**Structure**:
```
frontend/
├── src/
│   ├── pages/           # Page components
│   │   ├── Login.jsx
│   │   ├── TeacherDashboard.jsx
│   │   ├── StudentDashboard.jsx
│   │   ├── TeacherLectures.jsx
│   │   ├── TeacherLectureUpload.jsx
│   │   ├── TeacherProcessingStatus.jsx
│   │   └── SystemTest.jsx
│   ├── components/      # Reusable components
│   │   ├── Layout/
│   │   │   ├── Sidebar.jsx
│   │   │   └── Header.jsx
│   │   └── Common/
│   │       └── FileUpload.jsx
│   ├── services/        # API integration
│   │   ├── api.js       # Unified API client
│   │   └── lectureService.js  # Teacher Module client
│   ├── context/         # State management
│   │   └── AuthContext.jsx
│   ├── App.jsx          # Main application
│   └── main.jsx         # Entry point
├── package.json         # Dependencies
├── vite.config.js       # Build configuration
└── README.md
```

**Technology**:
- React 18.2.0
- Vite 5.4.21
- React Router 6.20.0
- Axios 1.6.2
- React Hot Toast
- Recharts

### 5. Database (`database/`)
**Purpose**: PostgreSQL schema and setup

**Files**:
- `schema.sql` - Complete database schema
- `README.md` - Database documentation

**Schema** (19 tables):
1. `users` - Base user table
2. `admins` - Admin profiles
3. `teachers` - Teacher profiles
4. `students` - Student profiles
5. `refresh_tokens` - JWT tokens
6. `lectures` - Lecture metadata
7. `sessions` - Teaching sessions
8. `session_participants` - Student participation
9. `face_recognition_data` - Student face data
10. `engagement_events` - Real-time engagement
11. `distraction_logs` - Distraction detection
12. `quizzes` - Quiz metadata
13. `quiz_questions` - MCQ questions
14. `quiz_attempts` - Student attempts
15. `quiz_answers` - Individual answers
16. `session_analytics` - Session metrics
17. `student_analytics` - Student metrics
18. `audit_logs` - System audit trail
19. `system_notifications` - User notifications

## Documentation Files

### Root Documentation
- **README.md** - Main project overview, features, quick start
- **LICENSE** - MIT License
- **CONTRIBUTING.md** - Contribution guidelines
- **START_SERVERS.md** - How to start all 3 servers
- **QUICK_FIX.md** - PostgreSQL troubleshooting

### Module-Specific READMEs
- `api/README.md` - Unified API documentation
- `classroom-ai-backend/README.md` - Teacher Module documentation
- `student-module/README.md` - Student Module documentation
- `frontend/README.md` - Frontend documentation
- `database/README.md` - Database documentation

## Configuration Files

### Python
- `requirements.txt` - Python dependencies (in each module)
- `.gitignore` - Ignore rules (excludes venv, models, logs, etc.)

### JavaScript
- `package.json` - Node.js dependencies
- `vite.config.js` - Vite build configuration

### Database
- `.env` files - Environment variables (not committed)
- `docker-compose.yml` - Docker services setup

## Excluded from Repository

The following are excluded via `.gitignore`:

**Development**:
- `venv/`, `.venv/`, `node_modules/`
- `__pycache__/`, `*.pyc`
- `.env`, `.env.local`
- `.claude/`

**Build Artifacts**:
- `dist/`, `build/`
- `*.egg-info/`

**AI Models & Cache**:
- `*.pth`, `*.pt`, `*.ckpt`
- `.model_cache/`
- `psych_book_faiss_index/`

**Data & Logs**:
- `logs/`, `*.log`
- `reports/`
- `testing/outputs/`
- `*.db`, `*.sqlite`

**Media**:
- `*.mp4`, `*.mp3`, `*.wav`
- `*.pdf`

**Documentation (Development)**:
- `*SESSION_SUMMARY.md`
- `*OPTIMIZATION*.md`
- `*TROUBLESHOOTING.md`

## Port Assignments

| Service | Port | Status |
|---------|------|--------|
| Unified API | 8002 | Production Ready |
| Teacher Module | 8001 | Production Ready |
| Student Module | 8001 | Production Ready |
| Frontend Dev | 3000 | Development |
| PostgreSQL | 5432 | Production Ready |

## Test Credentials

After running `api/setup_db.py`:
- Teacher: `teacher@example.com` / `password123`
- Student: `student@example.com` / `password123`

## Deployment Notes

### Development
1. Start PostgreSQL (Docker or local)
2. Initialize database: `python api/setup_db.py`
3. Start Unified API: `python api/main.py`
4. Start Teacher Module: `python classroom-ai-backend/main.py`
5. Start Frontend: `cd frontend && npm run dev`

### Production
- Use Gunicorn/Uvicorn workers for APIs
- Build frontend: `npm run build`
- Serve via Nginx
- Use environment variables for secrets
- Enable HTTPS
- Configure CORS properly
- Use PostgreSQL connection pooling

## Key Features

### For Teachers
- Upload lectures (audio/video)
- Automatic transcription (Whisper)
- AI-generated notes
- Auto-generated quizzes
- Content alignment with textbook
- Session management
- Student analytics

### For Students
- Live engagement monitoring
- Emotion detection
- Distraction alerts
- Quiz participation
- Performance tracking

## Technology Summary

**Backend**: Python 3.11+, FastAPI, SQLAlchemy, PostgreSQL
**Frontend**: React 18, Vite, React Router, Axios
**AI/ML**: PyTorch, TensorFlow, Transformers, Whisper, Llama 3.2, SBERT
**Authentication**: JWT, Bcrypt
**Database**: PostgreSQL 14+
**Deployment**: Docker, Nginx (production)

## Repository Size

Excluding:
- Virtual environments (~2GB)
- AI models (~3GB)
- Node modules (~500MB)
- Generated data

Core codebase: ~50MB

## Last Updated
2025-11-19

## Status
✅ Production Ready
