# EduVision Database Documentation

**PostgreSQL Schema for AI-Powered Educational Platform**

---

## Quick Links

- 📋 [User Stories & Requirements](USER_STORIES.md)
- 🗄️ [Database Schema (SQL)](schema.sql)
- 📖 [Database Design Documentation](DATABASE_DESIGN.md)

---

## Overview

This directory contains the complete database design for the EduVision platform, supporting:

- **Teacher Module** (Port 8000): Lecture creation, session management, analytics
- **Student Module** (Port 8001): Engagement tracking, face recognition, quiz participation
- **Admin Dashboard**: User management, platform monitoring, system analytics

---

## Key Features

### 🔐 Authentication & Authorization
- **JWT-based authentication** with access and refresh tokens
- **Role-based access control** (Admin, Teacher, Student)
- **Bcrypt password hashing** (cost factor 12)
- **Token revocation** mechanism

### 👥 User Management
- **Base `users` table** with email/password
- **Role-specific profiles**: `admins`, `teachers`, `students`
- **Face recognition data** storage for student identification
- **Audit logging** for all user actions

### 📚 Lecture & Session Management
- **Lectures**: Video, PDF, transcripts, textbook paragraphs
- **Sessions**: Live or recorded instances with join codes
- **Session participants**: Track attendance and engagement
- **Quiz system**: Questions, attempts, answers, grading

### 📊 Real-Time Engagement Tracking
- **engagement_events**: High-frequency data stream (every 1-2 seconds)
  - Face detection & recognition
  - Emotion analysis (7 types)
  - Head pose & gaze tracking
  - Concentration index
- **distraction_logs**: Discrete events with start/end times
  - Looking away, no face, yawning, etc.
- **Analytics aggregation**: Session and student-level summaries

---

## Database Schema Overview

### Core Tables (17 total)

#### User & Authentication
- `users` - Base authentication table
- `admins` - Admin profiles
- `teachers` - Teacher profiles
- `students` - Student profiles
- `face_recognition_data` - Face images and encodings
- `refresh_tokens` - JWT refresh token storage

#### Lectures & Sessions
- `lectures` - Lecture content (video, PDF, metadata)
- `sessions` - Live/recorded session instances
- `session_participants` - Attendance tracking

#### Engagement & Analytics
- `engagement_events` - Real-time student metrics
- `distraction_logs` - Distraction events
- `session_analytics` - Aggregated session metrics
- `student_analytics` - Per-student session metrics

#### Quizzes
- `quizzes` - Quiz metadata
- `quiz_questions` - Individual questions
- `quiz_attempts` - Student submissions
- `quiz_answers` - Answer records

#### System
- `audit_logs` - User action logging
- `system_notifications` - User notifications

---

## Quick Start

### 1. Prerequisites

```bash
# Install PostgreSQL 14+
sudo apt install postgresql postgresql-contrib

# Install Python dependencies (for application layer)
pip install psycopg2-binary sqlalchemy alembic
```

### 2. Create Database

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Create database and user
CREATE DATABASE eduvision;
CREATE USER eduvision_app WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE eduvision TO eduvision_app;
```

### 3. Run Schema

```bash
# Apply schema
psql -U eduvision_app -d eduvision -f schema.sql

# Verify tables created
psql -U eduvision_app -d eduvision -c "\dt"
```

### 4. Test Default Admin

```sql
-- Default admin credentials (CHANGE IN PRODUCTION!)
-- Email: admin@eduvision.com
-- Password: admin123

SELECT * FROM users WHERE role = 'admin';
SELECT * FROM admins;
```

---

## User Stories Summary

### Admin Stories
- **ADMIN-001**: User Management (CRUD operations)
- **ADMIN-002**: Platform Monitoring (system health, metrics)
- **ADMIN-003**: Lecture Management (view all lectures)
- **ADMIN-004**: Analytics Dashboard (platform-wide reports)

### Teacher Stories
- **TEACHER-001**: Account Management (login, profile)
- **TEACHER-002**: Lecture Creation (upload video/PDF)
- **TEACHER-003**: Live Lecture Session (start, monitor, end)
- **TEACHER-004**: Real-Time Monitoring (student engagement)
- **TEACHER-005**: Post-Lecture Analytics (engagement reports)
- **TEACHER-006**: Quiz Management (create, grade quizzes)

### Student Stories
- **STUDENT-001**: Account Registration & Login
- **STUDENT-002**: Join Lecture Session (via session code)
- **STUDENT-003**: Camera-Based Engagement Tracking
- **STUDENT-004**: Quiz Participation (answer questions)
- **STUDENT-005**: Personal Analytics Dashboard
- **STUDENT-006**: Privacy Controls (camera, data)

---

## Data Flow Examples

### Example 1: Teacher Creates Lecture

```
1. Teacher uploads video + PDF via Teacher Module API
2. API processes video → extracts transcript (Whisper)
3. API processes PDF → extracts paragraphs (PyMuPDF)
4. INSERT INTO lectures:
   - video_url
   - textbook_pdf_url
   - transcript_text (JSONB)
   - textbook_paragraphs (JSONB)
5. Return lecture_id to frontend
```

### Example 2: Student Joins Session

```
1. Student enters session_code on frontend
2. API validates code → SELECT * FROM sessions WHERE session_code = ?
3. INSERT INTO session_participants (session_id, student_id)
4. Activate Student Module camera
5. Start streaming engagement_events:
   - Every 1-2 seconds: INSERT INTO engagement_events
   - Face detected? identity_verified? emotion? gaze?
6. If distraction detected > 2 seconds:
   - INSERT INTO distraction_logs
```

### Example 3: Generate Analytics

```
1. Teacher ends session → UPDATE sessions SET status='completed'
2. Background job triggered:
   a. Aggregate engagement_events → session_analytics
   b. Aggregate per student → student_analytics
3. Teacher views analytics dashboard
4. Student views personal performance
```

---

## Key Design Decisions

### Why UUID Primary Keys?
- **Distributed systems**: No ID collision across microservices
- **Security**: Harder to enumerate records
- **Merging**: Easy to merge data from multiple sources

### Why JSONB for Metadata?
- **Flexibility**: Store complex data without schema changes
- **Performance**: Indexed JSONB queries are fast in PostgreSQL
- **Example**: `textbook_paragraphs`, `emotion_distribution`, `permissions`

### Why Partitioning?
- **Scalability**: `engagement_events` can have millions of rows
- **Performance**: Queries only scan relevant partitions
- **Archival**: Drop old partitions instead of deleting rows

### Why Separate Analytics Tables?
- **Performance**: Pre-aggregated data for dashboards
- **Caching**: Avoid re-calculating metrics on every request
- **Historical**: Preserve snapshots even if raw data is archived

---

## Performance Optimizations

### Indexes Created

**High-Priority**:
- `users(email)` - Login queries
- `sessions(session_code)` - Student joins
- `engagement_events(session_id, student_id, timestamp)` - Real-time queries
- `student_analytics(student_id)` - Student dashboards

**Medium-Priority**:
- `lectures(teacher_id)` - Teacher lecture list
- `quiz_attempts(student_id, quiz_id)` - Quiz history

### Connection Pooling

```python
# SQLAlchemy configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=20,        # Reuse 20 connections
    max_overflow=10,     # Allow 10 extra on demand
    pool_pre_ping=True   # Verify connection before use
)
```

### Query Optimization Tips

1. **Use indexes**: Always filter on indexed columns
2. **Limit results**: Add `LIMIT` to queries
3. **Batch inserts**: Use `INSERT INTO ... VALUES (...), (...), (...)`
4. **Avoid N+1**: Use `JOIN` or `IN` instead of loops
5. **Monitor slow queries**: Enable `pg_stat_statements`

---

## Security Checklist

- [x] Passwords hashed with bcrypt (cost factor 12)
- [x] JWT tokens with expiration (24h access, 7d refresh)
- [x] Role-based access control (RBAC)
- [x] SQL injection prevention (parameterized queries)
- [x] Face data encrypted at rest (pgcrypto)
- [x] Audit logging enabled
- [x] Default admin password documented (CHANGE IT!)
- [ ] SSL/TLS for database connections (configure in production)
- [ ] Regular backups scheduled
- [ ] Row-level security (RLS) policies (optional)

---

## Next Steps

### Implementation Roadmap

1. ✅ **Database Schema Design** - COMPLETED
2. ✅ **User Stories Documentation** - COMPLETED
3. ⏳ **SQLAlchemy Models** - NEXT
4. ⏳ **Alembic Migrations** - NEXT
5. ⏳ **API Integration** (Teacher Module)
6. ⏳ **API Integration** (Student Module)
7. ⏳ **Admin Dashboard Backend**
8. ⏳ **Testing & Performance Tuning**

### Files to Create Next

```
database/
├── models.py          # SQLAlchemy ORM models
├── migrations/        # Alembic migration scripts
│   └── versions/
├── connection.py      # Database connection config
├── crud.py            # CRUD operations
└── auth.py            # Authentication helpers
```

---

## Maintenance

### Daily Tasks
- Monitor slow query log
- Check disk space usage
- Verify backups completed

### Weekly Tasks
- `VACUUM ANALYZE` tables
- Review audit logs
- Check connection pool usage

### Monthly Tasks
- `VACUUM FULL` (requires downtime)
- Archive old engagement_events partitions
- Update PostgreSQL (minor versions)
- Security audit

---

## Support & Contact

**Documentation**: All files in this directory
**Schema Version**: 1.0
**Last Updated**: 2025-11-14
**Database**: PostgreSQL 14+

---

## License

Internal project documentation for EduVision graduation project.
