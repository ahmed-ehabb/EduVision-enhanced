# 🎓 Classroom AI System

**Advanced AI-Powered Educational Platform for Real-time Learning Analytics**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![React](https://img.shields.io/badge/React-18+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🚀 Quick Start

Get the entire system running with one command:

```bash
python run_classroom_ai.py
```

**That's it!** The script will:

- ✅ Check system requirements
- ✅ Start backend (6 AI models)
- ✅ Start frontend (React app)
- ✅ Open your browser automatically

Visit **http://localhost:3000** to access the system.

---

## 🎯 What This System Does

### 🎤 AI-Powered Audio Processing

- **Speech Recognition**: Convert lectures to accurate transcripts
- **Smart Translation**: Arabic ↔ English with context awareness
- **Auto Notes**: AI-generated summaries and key concepts
- **Content Alignment**: Match lectures to curriculum standards
- **Engagement Analysis**: Monitor student attention through audio patterns

### 👥 Multi-User Platform

- **Teachers**: Upload lectures, generate quizzes, monitor students
- **Students**: Access notes, take AI-generated quizzes, track progress
- **Admins**: System monitoring, user management, analytics

### 🔴 Real-Time Features

- **Live Sessions**: Real-time classroom monitoring
- **Face Recognition**: Automated attendance tracking
- **WebSocket Updates**: Live progress and engagement metrics

---

## 📋 System Requirements

- **Python 3.11+** ([Download](https://www.python.org/downloads/))
- **Node.js 18+** ([Download](https://nodejs.org/))
- **8GB RAM** (16GB recommended)
- **10GB Storage** for AI models and data

---

## 🛠 Installation

### Option 1: Automatic (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd classroom-ai-backend

# Run the unified startup script
python run_classroom_ai.py
```

### Option 2: Manual Setup

```bash
# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd eduvision-frontend
npm install
cd ..

# Start backend (Terminal 1)
python main.py

# Start frontend (Terminal 2)
cd eduvision-frontend
npm start
```

---

## 🎯 Core Features

### 🧠 6 AI Models Working Together

1. **ASR Model** - High-accuracy speech recognition
2. **Translation Model** - Bidirectional Arabic/English translation
3. **Notes Generator** - Educational content summarization
4. **Text Alignment** - Curriculum alignment analysis
5. **Engagement Analyzer** - Audio-based engagement detection
6. **Quiz Generator** - AI-powered question creation

### 📊 Comprehensive Analytics

- Real-time student engagement monitoring
- Attendance tracking with face recognition
- Language compliance analysis
- Performance metrics and reporting

### 🔒 Security & Privacy

- Firebase authentication
- GDPR-compliant data handling
- User consent management
- Encrypted sensitive data

---

## 🌐 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Models     │
│   React (3000)  │◄──►│   FastAPI       │◄──►│   6 Models      │
│   - Dashboards  │    │   (8001)        │    │   - ASR         │
│   - Components  │    │   - API Routes  │    │   - Translation │
│   - Real-time   │    │   - WebSockets  │    │   - Notes Gen   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▲
                                │
                       ┌─────────────────┐
                       │   Database      │
                       │   SQLite/       │
                       │   PostgreSQL    │
                       └─────────────────┘
```

---

## 📁 Project Structure

```
classroom-ai-backend/
├── 🚀 run_classroom_ai.py     # Main startup script
├── 📖 README.md               # This file
├── 📋 DEPLOYMENT_GUIDE.md     # Detailed setup guide
├── 📦 requirements.txt        # Python dependencies
├── 🔧 main.py                 # Backend entry point
├── 🤖 backend/                # AI models and API
│   ├── api_server.py         # FastAPI application
│   ├── database_enhanced.py  # Database management
│   ├── asr_module.py         # Speech recognition
│   ├── translation_module.py # Translation
│   ├── notes_generator.py    # Notes generation
│   ├── quiz_generator.py     # Quiz creation
│   ├── engagement_analyzer.py# Engagement analysis
│   └── text_alignment.py     # Content alignment
├── 🎨 eduvision-frontend/     # React application
│   ├── src/components/       # UI components
│   ├── src/services/         # API integration
│   └── package.json          # Frontend dependencies
└── 👥 facedataset/            # Face recognition data
```

---

## 🎯 Usage Examples

### For Teachers

```bash
1. Sign up at http://localhost:3000
2. Upload lecture audio files
3. Get AI analysis (transcript, notes, engagement)
4. Generate quizzes automatically
5. Monitor student performance
```

### For Students

```bash
1. Join courses with course codes
2. Access AI-generated lecture notes
3. Take automatically created quizzes
4. Track your engagement metrics
```

### For Administrators

```bash
1. Monitor system health and performance
2. Manage users and permissions
3. View comprehensive analytics
4. Configure system settings
```

---

## 🔧 Troubleshooting

### Common Issues

**Backend won't start:**

```bash
pip install pandas sentence-transformers PyJWT nltk
python main.py
```

**Frontend won't start:**

```bash
cd eduvision-frontend
npm install
npm start
```

**Missing dependencies:**

```bash
pip install -r requirements.txt
```

### System Health Check

- Backend API: http://localhost:8001/status
- API Documentation: http://localhost:8001/docs
- Frontend: http://localhost:3000

---

## 🌟 What's New

This cleaned-up version includes:

- 🚀 **Unified Startup**: One script to run everything
- 🧹 **Clean Architecture**: Consolidated codebase
- 🔧 **Better Error Handling**: Improved stability
- 📱 **Modern UI**: Enhanced user experience
- 🎯 **Performance Optimized**: Better resource management

---

## 📞 Support

### Quick Help

1. Check the [Deployment Guide](DEPLOYMENT_GUIDE.md) for detailed setup
2. Verify all dependencies are installed
3. Restart the system: `python run_classroom_ai.py`
4. Check system status endpoints

### Documentation

- **API Docs**: http://localhost:8001/docs
- **System Status**: http://localhost:8001/status
- **User Guides**: Check `docs/` directory

---

## 🎉 Ready to Use!

The Classroom AI system is now cleaned up, organized, and ready for deployment.

**Start with:** `python run_classroom_ai.py`

**Access at:** http://localhost:3000

---

_Built with ❤️ for educational innovation_
