# EduVision - Server Startup Guide

## Required Services

EduVision requires **3 separate services** to run:

1. **Unified API** (Port 8002) - Authentication, Lectures, Sessions
2. **Teacher Module API** (Port 8000) - AI Processing, File Upload
3. **Frontend Dev Server** (Port 3000) - React Application

---

## Quick Start (All Services)

### Terminal 1: Unified API
```bash
cd api
python main.py
```
**Expected Output:** `Uvicorn running on http://0.0.0.0:8002`

### Terminal 2: Teacher Module API
```bash
cd teacher_module
python run.py
```
**Expected Output:** `Uvicorn running on http://0.0.0.0:8000`

### Terminal 3: Frontend
```bash
cd frontend
npm run dev
```
**Expected Output:** `Local: http://localhost:3000/`

---

## Detailed Setup

### 1. Database Setup (First Time Only)

Make sure PostgreSQL is running and the database is created:

```bash
cd api
python setup_db.py
```

This creates:
- Database tables (19 tables)
- Sample test users
- Initial data

### 2. Test Credentials

After running `setup_db.py`, you can login with:

**Teacher Account:**
- Email: `teacher@example.com`
- Password: `password123`

**Student Account:**
- Email: `student@example.com`
- Password: `password123`

---

## Service Details

### Unified API (Port 8002)

**Purpose:** Main backend API for authentication, lecture metadata, and sessions

**Endpoints:**
- `POST /auth/login` - User login
- `GET /auth/me` - Get current user
- `GET /api/lectures` - List lectures
- `POST /api/lectures` - Create lecture metadata
- `GET /api/sessions` - List sessions
- `POST /api/sessions` - Create session

**Tech Stack:** FastAPI, SQLAlchemy, PostgreSQL, JWT

### Teacher Module API (Port 8000)

**Purpose:** AI processing for lectures (transcription, translation, notes, quizzes)

**Endpoints:**
- `POST /api/lectures/upload` - Upload lecture audio/video
- `GET /api/lectures/{job_id}/status` - Check processing status
- `GET /api/lectures/{job_id}/transcript` - Get transcript
- `GET /api/lectures/{job_id}/notes` - Get AI-generated notes
- `GET /api/lectures/{job_id}/quiz` - Get AI-generated quiz

**AI Models:**
- ASR (Automatic Speech Recognition)
- Arabic ↔ English Translation
- Notes Generation (GPT-based)
- Quiz Generation
- Content Alignment with Textbook

### Frontend (Port 3000)

**Purpose:** React-based user interface

**Tech Stack:** React 18, Vite, React Router, Axios

**Features:**
- Teacher Dashboard
- Lecture Management (List, Upload, Process)
- Session Management
- System Test Page (NEW!)

---

## Troubleshooting

### Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Windows - Kill process on port
netstat -ano | findstr :8002
taskkill /PID <PID> /F

# Or change port in config files
```

### Database Connection Error

**Error:** `could not connect to server`

**Solution:**
1. Check PostgreSQL is running
2. Verify credentials in `api/.env`
3. Check database exists: `psql -U postgres -c "\l"`

### Module Not Found

**Error:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```bash
# Install Python dependencies
cd api
pip install -r requirements.txt

cd ../teacher_module
pip install -r requirements.txt

# Install Node dependencies
cd ../frontend
npm install
```

---

## API Testing

### Using the System Test Page

1. Start all 3 services
2. Login to frontend: http://localhost:3000
3. Navigate to "System Test" in sidebar
4. Click "Run All Tests"

This will test:
- ✓ Unified API connectivity
- ✓ Teacher Module API connectivity
- ✓ Authentication
- ✓ Lecture endpoints
- ✓ Session endpoints

### Using curl

**Test Unified API:**
```bash
curl http://localhost:8002/health
```

**Test Teacher Module API:**
```bash
curl http://localhost:8000/health
```

---

## Development Workflow

### Full Stack Development

1. **Backend Changes:**
   - Edit files in `api/` or `teacher_module/`
   - FastAPI auto-reloads on file changes
   - Check logs in terminal

2. **Frontend Changes:**
   - Edit files in `frontend/src/`
   - Vite HMR (Hot Module Replacement) auto-updates browser
   - Check browser console for errors

3. **Database Changes:**
   - Update models in `api/models.py`
   - Run migrations or recreate: `python setup_db.py`

---

## Production Deployment

**Not yet configured.** For production, you'll need:

1. Use production WSGI server (Gunicorn)
2. Build frontend: `npm run build`
3. Serve built files via Nginx
4. Use production database
5. Set environment variables
6. Enable HTTPS
7. Configure CORS properly

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                     │
│                   http://localhost:3000                 │
└────────────┬──────────────────────────┬─────────────────┘
             │                          │
             │                          │
    ┌────────▼────────┐        ┌────────▼──────────┐
    │   Unified API   │        │  Teacher Module   │
    │   Port 8002     │        │    Port 8000      │
    │                 │        │                   │
    │  - Auth         │        │  - File Upload    │
    │  - Lectures     │        │  - AI Processing  │
    │  - Sessions     │        │  - Transcription  │
    └────────┬────────┘        │  - Translation    │
             │                 │  - Notes/Quiz     │
             │                 └───────────────────┘
    ┌────────▼────────┐
    │   PostgreSQL    │
    │   Port 5432     │
    └─────────────────┘
```

---

## Environment Variables

### api/.env
```
DATABASE_URL=postgresql://postgres:password@localhost:5432/eduvision
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=1440
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### teacher_module/.env
```
OPENAI_API_KEY=your-openai-key-here
DATABASE_URL=postgresql://postgres:password@localhost:5432/eduvision
```

### frontend/.env
```
VITE_API_URL=http://localhost:8002
VITE_TEACHER_API_URL=http://localhost:8000
```

---

## Next Steps

After starting all servers:

1. ✅ Login at http://localhost:3000
2. ✅ Go to "System Test" page
3. ✅ Run all tests to verify connectivity
4. ✅ Try creating a test lecture
5. ✅ Upload a lecture file (if you have audio/video)
6. ✅ Monitor processing status

**Happy coding!** 🚀
