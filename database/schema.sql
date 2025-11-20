-- ============================================================================
-- EduVision Database Schema - PostgreSQL
-- ============================================================================
-- Version: 1.0
-- Date: 2025-11-14
-- Description: Comprehensive schema for Teacher Module, Student Module, and Admin Dashboard
-- ============================================================================

-- Enable UUID extension for primary keys
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pgcrypto for password hashing
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- ENUMS
-- ============================================================================

CREATE TYPE user_role AS ENUM ('admin', 'teacher', 'student');
CREATE TYPE session_status AS ENUM ('scheduled', 'active', 'completed', 'cancelled');
CREATE TYPE distraction_type AS ENUM ('no_face', 'looking_away', 'yawning', 'hand_to_face', 'phone_usage', 'low_blink_rate');
CREATE TYPE emotion_type AS ENUM ('angry', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'disgust');
CREATE TYPE quiz_question_type AS ENUM ('multiple_choice', 'true_false', 'short_answer');

-- ============================================================================
-- TABLE: users (Base table for all user types)
-- ============================================================================

CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role user_role NOT NULL,
    is_active BOOLEAN DEFAULT true,
    email_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,

    CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_active ON users(is_active);

-- ============================================================================
-- TABLE: admins
-- ============================================================================

CREATE TABLE admins (
    admin_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID UNIQUE NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    full_name VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    permissions JSONB DEFAULT '{"can_create_users": true, "can_delete_users": true, "can_view_analytics": true, "can_manage_lectures": true}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_admins_user_id ON admins(user_id);

-- ============================================================================
-- TABLE: teachers
-- ============================================================================

CREATE TABLE teachers (
    teacher_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID UNIQUE NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    full_name VARCHAR(255) NOT NULL,
    department VARCHAR(100),
    bio TEXT,
    profile_image_url VARCHAR(500),
    phone VARCHAR(20),
    office_hours VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_teachers_user_id ON teachers(user_id);
CREATE INDEX idx_teachers_department ON teachers(department);

-- ============================================================================
-- TABLE: students
-- ============================================================================

CREATE TABLE students (
    student_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID UNIQUE NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    full_name VARCHAR(255) NOT NULL,
    student_number VARCHAR(50) UNIQUE,
    major VARCHAR(100),
    year_of_study INTEGER,
    profile_image_url VARCHAR(500),
    phone VARCHAR(20),
    date_of_birth DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT year_valid CHECK (year_of_study BETWEEN 1 AND 6)
);

CREATE INDEX idx_students_user_id ON students(user_id);
CREATE INDEX idx_students_number ON students(student_number);
CREATE INDEX idx_students_major ON students(major);

-- ============================================================================
-- TABLE: face_recognition_data (For student face recognition)
-- ============================================================================

CREATE TABLE face_recognition_data (
    face_data_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id UUID NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    image_url VARCHAR(500) NOT NULL,
    image_hash VARCHAR(64) NOT NULL, -- SHA256 hash for deduplication
    encoding_data BYTEA, -- Stored face encoding (encrypted)
    is_active BOOLEAN DEFAULT true,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_image_per_student UNIQUE(student_id, image_hash)
);

CREATE INDEX idx_face_data_student ON face_recognition_data(student_id);
CREATE INDEX idx_face_data_active ON face_recognition_data(is_active);

-- ============================================================================
-- TABLE: refresh_tokens (For JWT authentication)
-- ============================================================================

CREATE TABLE refresh_tokens (
    token_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    revoked BOOLEAN DEFAULT false,
    revoked_at TIMESTAMP WITH TIME ZONE,
    user_agent VARCHAR(500),
    ip_address INET
);

CREATE INDEX idx_refresh_tokens_user ON refresh_tokens(user_id);
CREATE INDEX idx_refresh_tokens_hash ON refresh_tokens(token_hash);
CREATE INDEX idx_refresh_tokens_expires ON refresh_tokens(expires_at);

-- ============================================================================
-- TABLE: lectures
-- ============================================================================

CREATE TABLE lectures (
    lecture_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    teacher_id UUID NOT NULL REFERENCES teachers(teacher_id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    course_code VARCHAR(50), -- e.g., "CS401"
    course_name VARCHAR(255), -- e.g., "Machine Learning"
    video_url VARCHAR(500),
    video_duration INTEGER, -- in seconds
    textbook_pdf_url VARCHAR(500),
    thumbnail_url VARCHAR(500),
    chapter_start_anchor TEXT,
    chapter_end_anchor TEXT,
    transcript_text TEXT, -- Generated from video
    textbook_paragraphs JSONB, -- Extracted paragraphs from PDF
    is_published BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    published_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_lectures_teacher ON lectures(teacher_id);
CREATE INDEX idx_lectures_published ON lectures(is_published);
CREATE INDEX idx_lectures_created ON lectures(created_at DESC);
CREATE INDEX idx_lectures_course_code ON lectures(course_code);

-- ============================================================================
-- TABLE: sessions (Live or recorded lecture sessions)
-- ============================================================================

CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lecture_id UUID NOT NULL REFERENCES lectures(lecture_id) ON DELETE CASCADE,
    teacher_id UUID NOT NULL REFERENCES teachers(teacher_id) ON DELETE CASCADE,
    session_code VARCHAR(20) UNIQUE NOT NULL, -- Join code for students
    status session_status DEFAULT 'scheduled',
    scheduled_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE,
    duration INTEGER, -- Actual duration in seconds
    total_students INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_lecture ON sessions(lecture_id);
CREATE INDEX idx_sessions_teacher ON sessions(teacher_id);
CREATE INDEX idx_sessions_code ON sessions(session_code);
CREATE INDEX idx_sessions_status ON sessions(status);
CREATE INDEX idx_sessions_scheduled ON sessions(scheduled_at);

-- ============================================================================
-- TABLE: session_participants
-- ============================================================================

CREATE TABLE session_participants (
    participant_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    left_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    was_camera_enabled BOOLEAN DEFAULT false,
    was_face_recognized BOOLEAN DEFAULT false,

    CONSTRAINT unique_student_session UNIQUE(session_id, student_id)
);

CREATE INDEX idx_participants_session ON session_participants(session_id);
CREATE INDEX idx_participants_student ON session_participants(student_id);
CREATE INDEX idx_participants_joined ON session_participants(joined_at);

-- ============================================================================
-- TABLE: engagement_events (Real-time engagement tracking)
-- ============================================================================

CREATE TABLE engagement_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    video_timestamp INTEGER, -- Position in lecture video (seconds)

    -- Engagement metrics
    face_detected BOOLEAN DEFAULT false,
    identity_verified BOOLEAN DEFAULT false,
    emotion emotion_type,
    emotion_confidence DECIMAL(3, 2), -- 0.00 to 1.00

    -- Head pose
    head_pitch DECIMAL(5, 2), -- Degrees
    head_yaw DECIMAL(5, 2),
    head_roll DECIMAL(5, 2),

    -- Eye metrics
    gaze_ratio DECIMAL(3, 2),
    gaze_direction VARCHAR(10), -- 'LEFT', 'RIGHT', 'CENTER'
    ear DECIMAL(3, 2), -- Eye Aspect Ratio

    -- Concentration
    concentration_index VARCHAR(20), -- 'ENGAGED', 'DISTRACTED', 'DROWSY', etc.

    -- Raw data (for debugging)
    raw_data JSONB
);

CREATE INDEX idx_engagement_session ON engagement_events(session_id);
CREATE INDEX idx_engagement_student ON engagement_events(student_id);
CREATE INDEX idx_engagement_timestamp ON engagement_events(timestamp);
CREATE INDEX idx_engagement_video_ts ON engagement_events(video_timestamp);

-- Partition by month for performance
-- CREATE TABLE engagement_events_y2025m01 PARTITION OF engagement_events
--   FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- ============================================================================
-- TABLE: distraction_logs
-- ============================================================================

CREATE TABLE distraction_logs (
    distraction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    distraction_type distraction_type NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    video_timestamp INTEGER, -- Position in lecture video
    severity INTEGER CHECK (severity BETWEEN 1 AND 5), -- 1 = low, 5 = high
    details JSONB -- Additional context
);

CREATE INDEX idx_distraction_session ON distraction_logs(session_id);
CREATE INDEX idx_distraction_student ON distraction_logs(student_id);
CREATE INDEX idx_distraction_type ON distraction_logs(distraction_type);
CREATE INDEX idx_distraction_started ON distraction_logs(started_at);

-- ============================================================================
-- TABLE: quizzes
-- ============================================================================

CREATE TABLE quizzes (
    quiz_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lecture_id UUID NOT NULL REFERENCES lectures(lecture_id) ON DELETE CASCADE,
    teacher_id UUID NOT NULL REFERENCES teachers(teacher_id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    video_timestamp INTEGER NOT NULL, -- When quiz should appear in video
    time_limit_seconds INTEGER, -- NULL = no time limit
    passing_score DECIMAL(5, 2), -- Percentage (0.00 to 100.00)
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_quizzes_lecture ON quizzes(lecture_id);
CREATE INDEX idx_quizzes_teacher ON quizzes(teacher_id);
CREATE INDEX idx_quizzes_active ON quizzes(is_active);

-- ============================================================================
-- TABLE: quiz_questions
-- ============================================================================

CREATE TABLE quiz_questions (
    question_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    quiz_id UUID NOT NULL REFERENCES quizzes(quiz_id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    question_type quiz_question_type NOT NULL,
    options JSONB, -- For multiple choice: ["Option A", "Option B", ...]
    correct_answer TEXT NOT NULL,
    points DECIMAL(5, 2) DEFAULT 1.00,
    explanation TEXT,
    order_index INTEGER NOT NULL,

    CONSTRAINT unique_question_order UNIQUE(quiz_id, order_index)
);

CREATE INDEX idx_questions_quiz ON quiz_questions(quiz_id);
CREATE INDEX idx_questions_order ON quiz_questions(quiz_id, order_index);

-- ============================================================================
-- TABLE: quiz_attempts
-- ============================================================================

CREATE TABLE quiz_attempts (
    attempt_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    quiz_id UUID NOT NULL REFERENCES quizzes(quiz_id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(session_id) ON DELETE SET NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    submitted_at TIMESTAMP WITH TIME ZONE,
    score DECIMAL(5, 2), -- Percentage
    total_points DECIMAL(5, 2),
    earned_points DECIMAL(5, 2),
    passed BOOLEAN,
    time_taken_seconds INTEGER
);

CREATE INDEX idx_attempts_quiz ON quiz_attempts(quiz_id);
CREATE INDEX idx_attempts_student ON quiz_attempts(student_id);
CREATE INDEX idx_attempts_session ON quiz_attempts(session_id);

-- ============================================================================
-- TABLE: quiz_answers
-- ============================================================================

CREATE TABLE quiz_answers (
    answer_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    attempt_id UUID NOT NULL REFERENCES quiz_attempts(attempt_id) ON DELETE CASCADE,
    question_id UUID NOT NULL REFERENCES quiz_questions(question_id) ON DELETE CASCADE,
    student_answer TEXT,
    is_correct BOOLEAN,
    points_earned DECIMAL(5, 2),
    answered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_answer_per_attempt UNIQUE(attempt_id, question_id)
);

CREATE INDEX idx_answers_attempt ON quiz_answers(attempt_id);
CREATE INDEX idx_answers_question ON quiz_answers(question_id);

-- ============================================================================
-- TABLE: session_analytics (Aggregated session-level metrics)
-- ============================================================================

CREATE TABLE session_analytics (
    analytics_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID UNIQUE NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,

    -- Participation
    total_students INTEGER DEFAULT 0,
    avg_attendance_duration_seconds INTEGER,

    -- Engagement
    avg_engagement_score DECIMAL(5, 2), -- 0.00 to 100.00
    total_distraction_events INTEGER DEFAULT 0,
    most_common_distraction distraction_type,

    -- Emotions
    emotion_distribution JSONB, -- {"happy": 45, "neutral": 30, ...}
    dominant_emotion emotion_type,

    -- Quiz performance
    total_quiz_attempts INTEGER DEFAULT 0,
    avg_quiz_score DECIMAL(5, 2),
    quiz_pass_rate DECIMAL(5, 2),

    -- Timestamps
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analytics_session ON session_analytics(session_id);

-- ============================================================================
-- TABLE: student_analytics (Aggregated student-level metrics)
-- ============================================================================

CREATE TABLE student_analytics (
    analytics_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id UUID NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,

    -- Attendance
    duration_seconds INTEGER,
    attendance_percentage DECIMAL(5, 2), -- % of session attended

    -- Engagement
    engagement_score DECIMAL(5, 2), -- 0.00 to 100.00
    total_distraction_count INTEGER DEFAULT 0,
    distraction_breakdown JSONB, -- {"no_face": 5, "looking_away": 12, ...}

    -- Emotions
    dominant_emotion emotion_type,
    emotion_distribution JSONB,
    emotion_changes_count INTEGER, -- Frequency of emotion shifts

    -- Focus metrics
    avg_concentration_index VARCHAR(20),
    focus_periods JSONB, -- [{"start": 120, "end": 240, "level": "ENGAGED"}, ...]

    -- Quiz performance
    quiz_score DECIMAL(5, 2),
    quiz_rank INTEGER, -- Rank among peers

    -- Generated data
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_student_session_analytics UNIQUE(student_id, session_id)
);

CREATE INDEX idx_student_analytics_student ON student_analytics(student_id);
CREATE INDEX idx_student_analytics_session ON student_analytics(session_id);

-- ============================================================================
-- TABLE: audit_logs (For admin monitoring and compliance)
-- ============================================================================

CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL, -- 'LOGIN', 'LOGOUT', 'CREATE_USER', 'DELETE_LECTURE', etc.
    resource_type VARCHAR(50), -- 'user', 'lecture', 'session', etc.
    resource_id UUID,
    ip_address INET,
    user_agent VARCHAR(500),
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_action ON audit_logs(action);
CREATE INDEX idx_audit_created ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_resource ON audit_logs(resource_type, resource_id);

-- ============================================================================
-- TABLE: system_notifications
-- ============================================================================

CREATE TABLE system_notifications (
    notification_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    type VARCHAR(50), -- 'info', 'warning', 'error', 'success'
    is_read BOOLEAN DEFAULT false,
    link_url VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    read_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_notifications_user ON system_notifications(user_id);
CREATE INDEX idx_notifications_read ON system_notifications(is_read);
CREATE INDEX idx_notifications_created ON system_notifications(created_at DESC);

-- ============================================================================
-- FUNCTIONS & TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_teachers_updated_at BEFORE UPDATE ON teachers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_students_updated_at BEFORE UPDATE ON students
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_lectures_updated_at BEFORE UPDATE ON lectures
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to hash passwords (use with application layer bcrypt instead)
CREATE OR REPLACE FUNCTION hash_password(plain_password TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN crypt(plain_password, gen_salt('bf', 12));
END;
$$ LANGUAGE plpgsql;

-- Function to verify passwords
CREATE OR REPLACE FUNCTION verify_password(plain_password TEXT, password_hash TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN password_hash = crypt(plain_password, password_hash);
END;
$$ LANGUAGE plpgsql;

-- Function to generate random session code
CREATE OR REPLACE FUNCTION generate_session_code()
RETURNS TEXT AS $$
DECLARE
    chars TEXT := 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; -- Excluding similar chars
    result TEXT := '';
    i INTEGER;
BEGIN
    FOR i IN 1..8 LOOP
        result := result || substr(chars, floor(random() * length(chars) + 1)::int, 1);
    END LOOP;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View: Active sessions with teacher and lecture info
CREATE OR REPLACE VIEW active_sessions_view AS
SELECT
    s.session_id,
    s.session_code,
    s.status,
    s.started_at,
    s.total_students,
    l.title AS lecture_title,
    t.full_name AS teacher_name,
    t.teacher_id
FROM sessions s
JOIN lectures l ON s.lecture_id = l.lecture_id
JOIN teachers t ON s.teacher_id = t.teacher_id
WHERE s.status = 'active';

-- View: Student engagement summary
CREATE OR REPLACE VIEW student_engagement_summary AS
SELECT
    s.student_id,
    s.full_name,
    COUNT(DISTINCT sp.session_id) AS sessions_attended,
    AVG(sa.engagement_score) AS avg_engagement_score,
    SUM(sa.total_distraction_count) AS total_distractions,
    AVG(sa.quiz_score) AS avg_quiz_score
FROM students s
LEFT JOIN session_participants sp ON s.student_id = sp.student_id
LEFT JOIN student_analytics sa ON s.student_id = sa.student_id
GROUP BY s.student_id, s.full_name;

-- View: Teacher lecture statistics
CREATE OR REPLACE VIEW teacher_lecture_stats AS
SELECT
    t.teacher_id,
    t.full_name,
    COUNT(DISTINCT l.lecture_id) AS total_lectures,
    COUNT(DISTINCT s.session_id) AS total_sessions,
    AVG(sa.avg_engagement_score) AS avg_class_engagement,
    SUM(s.total_students) AS total_students_taught
FROM teachers t
LEFT JOIN lectures l ON t.teacher_id = l.teacher_id
LEFT JOIN sessions s ON l.lecture_id = s.lecture_id
LEFT JOIN session_analytics sa ON s.session_id = sa.session_id
GROUP BY t.teacher_id, t.full_name;

-- ============================================================================
-- INITIAL DATA (Default Admin Account)
-- ============================================================================

-- Create default admin user (password: 'admin123' - CHANGE IN PRODUCTION!)
DO $$
DECLARE
    admin_user_id UUID;
BEGIN
    INSERT INTO users (email, password_hash, role, email_verified)
    VALUES ('admin@eduvision.com', hash_password('admin123'), 'admin', true)
    RETURNING user_id INTO admin_user_id;

    INSERT INTO admins (user_id, full_name, permissions)
    VALUES (admin_user_id, 'System Administrator', '{"can_create_users": true, "can_delete_users": true, "can_view_analytics": true, "can_manage_lectures": true}'::jsonb);
END $$;

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Composite indexes for common queries
CREATE INDEX idx_engagement_session_student ON engagement_events(session_id, student_id, timestamp);
CREATE INDEX idx_distraction_session_student_type ON distraction_logs(session_id, student_id, distraction_type);
CREATE INDEX idx_quiz_attempts_student_quiz ON quiz_attempts(student_id, quiz_id, submitted_at DESC);

-- ============================================================================
-- COMMENTS (Documentation)
-- ============================================================================

COMMENT ON TABLE users IS 'Base table for all user types (admin, teacher, student)';
COMMENT ON TABLE teachers IS 'Teacher-specific profile information';
COMMENT ON TABLE students IS 'Student-specific profile information';
COMMENT ON TABLE face_recognition_data IS 'Face images and encodings for student identification';
COMMENT ON TABLE lectures IS 'Lecture content including video, PDF, and transcripts';
COMMENT ON TABLE sessions IS 'Live or recorded lecture sessions';
COMMENT ON TABLE engagement_events IS 'Real-time student engagement metrics during sessions';
COMMENT ON TABLE distraction_logs IS 'Logged distraction events with timestamps and types';
COMMENT ON TABLE session_analytics IS 'Aggregated analytics per session';
COMMENT ON TABLE student_analytics IS 'Aggregated analytics per student per session';

-- ============================================================================
-- GRANT PERMISSIONS (Configure based on your needs)
-- ============================================================================

-- Create application role
-- CREATE ROLE eduvision_app WITH LOGIN PASSWORD 'your_secure_password';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO eduvision_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO eduvision_app;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
