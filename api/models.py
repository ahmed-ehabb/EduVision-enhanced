"""
SQLAlchemy ORM Models for EduVision Database
Maps to the PostgreSQL schema defined in database/schema.sql
"""

from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime, ForeignKey,
    Text, DECIMAL, Enum as SQLEnum, TIMESTAMP, JSON, LargeBinary, Date
)
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import enum
import uuid

# Create base class for declarative models
Base = declarative_base()

# ============================================================================
# ENUMS
# ============================================================================

class UserRole(str, enum.Enum):
    """User role types"""
    ADMIN = "admin"
    TEACHER = "teacher"
    STUDENT = "student"

class SessionStatus(str, enum.Enum):
    """Session status types"""
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class DistractionType(str, enum.Enum):
    """Types of distractions"""
    NO_FACE = "no_face"
    LOOKING_AWAY = "looking_away"
    YAWNING = "yawning"
    HAND_TO_FACE = "hand_to_face"
    PHONE_USAGE = "phone_usage"
    LOW_BLINK_RATE = "low_blink_rate"

class EmotionType(str, enum.Enum):
    """Emotion types"""
    ANGRY = "angry"
    FEAR = "fear"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"
    DISGUST = "disgust"

class QuizQuestionType(str, enum.Enum):
    """Quiz question types"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"

# ============================================================================
# MODELS
# ============================================================================

class User(Base):
    """Base user table for all user types"""
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole, name="user_role", values_callable=lambda x: [e.value for e in x]), nullable=False, index=True)
    is_active = Column(Boolean, default=True, index=True)
    email_verified = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    admin = relationship("Admin", back_populates="user", uselist=False, cascade="all, delete-orphan")
    teacher = relationship("Teacher", back_populates="user", uselist=False, cascade="all, delete-orphan")
    student = relationship("Student", back_populates="user", uselist=False, cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("SystemNotification", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")

    def __repr__(self):
        return f"<User(email='{self.email}', role='{self.role}')>"


class Admin(Base):
    """Admin user profiles"""
    __tablename__ = "admins"

    admin_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=True)
    permissions = Column(JSONB, default={
        "can_create_users": True,
        "can_delete_users": True,
        "can_view_analytics": True,
        "can_manage_lectures": True
    })
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="admin")

    def __repr__(self):
        return f"<Admin(name='{self.full_name}')>"


class Teacher(Base):
    """Teacher user profiles"""
    __tablename__ = "teachers"

    teacher_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    department = Column(String(100), nullable=True, index=True)
    bio = Column(Text, nullable=True)
    profile_image_url = Column(String(500), nullable=True)
    phone = Column(String(20), nullable=True)
    office_hours = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="teacher")
    lectures = relationship("Lecture", back_populates="teacher", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="teacher", cascade="all, delete-orphan")
    quizzes = relationship("Quiz", back_populates="teacher", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Teacher(name='{self.full_name}', department='{self.department}')>"


class Student(Base):
    """Student user profiles"""
    __tablename__ = "students"

    student_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    student_number = Column(String(50), unique=True, nullable=True, index=True)
    major = Column(String(100), nullable=True, index=True)
    year_of_study = Column(Integer, nullable=True)
    profile_image_url = Column(String(500), nullable=True)
    phone = Column(String(20), nullable=True)
    date_of_birth = Column(Date, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="student")
    face_recognition_data = relationship("FaceRecognitionData", back_populates="student", cascade="all, delete-orphan")
    session_participants = relationship("SessionParticipant", back_populates="student", cascade="all, delete-orphan")
    engagement_events = relationship("EngagementEvent", back_populates="student", cascade="all, delete-orphan")
    distraction_logs = relationship("DistractionLog", back_populates="student", cascade="all, delete-orphan")
    quiz_attempts = relationship("QuizAttempt", back_populates="student", cascade="all, delete-orphan")
    student_analytics = relationship("StudentAnalytics", back_populates="student", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Student(name='{self.full_name}', number='{self.student_number}')>"


class FaceRecognitionData(Base):
    """Face recognition data for students"""
    __tablename__ = "face_recognition_data"

    face_data_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.student_id", ondelete="CASCADE"), nullable=False, index=True)
    image_url = Column(String(500), nullable=False)
    image_hash = Column(String(64), nullable=False)  # SHA256 hash
    encoding_data = Column(LargeBinary, nullable=True)  # Encrypted face encoding
    is_active = Column(Boolean, default=True, index=True)
    uploaded_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    student = relationship("Student", back_populates="face_recognition_data")

    def __repr__(self):
        return f"<FaceRecognitionData(student_id='{self.student_id}')>"


class RefreshToken(Base):
    """JWT refresh tokens"""
    __tablename__ = "refresh_tokens"

    token_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(255), nullable=False, index=True)
    expires_at = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    revoked = Column(Boolean, default=False)
    revoked_at = Column(TIMESTAMP(timezone=True), nullable=True)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(INET, nullable=True)

    # Relationships
    user = relationship("User", back_populates="refresh_tokens")

    def __repr__(self):
        return f"<RefreshToken(user_id='{self.user_id}', revoked={self.revoked})>"


class Lecture(Base):
    """Lecture content and metadata"""
    __tablename__ = "lectures"

    lecture_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    teacher_id = Column(UUID(as_uuid=True), ForeignKey("teachers.teacher_id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    course_code = Column(String(50), nullable=True, index=True)  # e.g., "CS401"
    course_name = Column(String(255), nullable=True)  # e.g., "Machine Learning"
    video_url = Column(String(500), nullable=True)
    video_duration = Column(Integer, nullable=True)  # in seconds
    textbook_pdf_url = Column(String(500), nullable=True)
    thumbnail_url = Column(String(500), nullable=True)
    chapter_start_anchor = Column(Text, nullable=True)
    chapter_end_anchor = Column(Text, nullable=True)
    transcript_text = Column(Text, nullable=True)  # Generated from video
    textbook_paragraphs = Column(JSONB, nullable=True)  # Extracted from PDF
    is_published = Column(Boolean, default=False, index=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    published_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    teacher = relationship("Teacher", back_populates="lectures")
    sessions = relationship("Session", back_populates="lecture", cascade="all, delete-orphan")
    quizzes = relationship("Quiz", back_populates="lecture", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Lecture(title='{self.title}', teacher_id='{self.teacher_id}')>"


class Session(Base):
    """Live or recorded lecture sessions"""
    __tablename__ = "sessions"

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lecture_id = Column(UUID(as_uuid=True), ForeignKey("lectures.lecture_id", ondelete="CASCADE"), nullable=False, index=True)
    teacher_id = Column(UUID(as_uuid=True), ForeignKey("teachers.teacher_id", ondelete="CASCADE"), nullable=False, index=True)
    session_code = Column(String(20), unique=True, nullable=False, index=True)
    status = Column(SQLEnum(SessionStatus, name="session_status", values_callable=lambda x: [e.value for e in x]), default=SessionStatus.SCHEDULED, index=True)
    scheduled_at = Column(TIMESTAMP(timezone=True), nullable=True, index=True)
    started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    ended_at = Column(TIMESTAMP(timezone=True), nullable=True)
    duration = Column(Integer, nullable=True)  # in seconds
    total_students = Column(Integer, default=0)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    lecture = relationship("Lecture", back_populates="sessions")
    teacher = relationship("Teacher", back_populates="sessions")
    participants = relationship("SessionParticipant", back_populates="session", cascade="all, delete-orphan")
    engagement_events = relationship("EngagementEvent", back_populates="session", cascade="all, delete-orphan")
    distraction_logs = relationship("DistractionLog", back_populates="session", cascade="all, delete-orphan")
    quiz_attempts = relationship("QuizAttempt", back_populates="session")
    session_analytics = relationship("SessionAnalytics", back_populates="session", uselist=False, cascade="all, delete-orphan")
    student_analytics = relationship("StudentAnalytics", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Session(code='{self.session_code}', status='{self.status}')>"


class SessionParticipant(Base):
    """Students participating in sessions"""
    __tablename__ = "session_participants"

    participant_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.student_id", ondelete="CASCADE"), nullable=False, index=True)
    joined_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)
    left_at = Column(TIMESTAMP(timezone=True), nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    was_camera_enabled = Column(Boolean, default=False)
    was_face_recognized = Column(Boolean, default=False)

    # Relationships
    session = relationship("Session", back_populates="participants")
    student = relationship("Student", back_populates="session_participants")

    def __repr__(self):
        return f"<SessionParticipant(session_id='{self.session_id}', student_id='{self.student_id}')>"


class EngagementEvent(Base):
    """Real-time engagement tracking events"""
    __tablename__ = "engagement_events"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.student_id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)
    video_timestamp = Column(Integer, nullable=True, index=True)  # Position in lecture video

    # Engagement metrics
    face_detected = Column(Boolean, default=False)
    identity_verified = Column(Boolean, default=False)
    emotion = Column(SQLEnum(EmotionType, name="emotion_type", values_callable=lambda x: [e.value for e in x]), nullable=True)
    emotion_confidence = Column(DECIMAL(3, 2), nullable=True)  # 0.00 to 1.00

    # Head pose
    head_pitch = Column(DECIMAL(5, 2), nullable=True)
    head_yaw = Column(DECIMAL(5, 2), nullable=True)
    head_roll = Column(DECIMAL(5, 2), nullable=True)

    # Eye metrics
    gaze_ratio = Column(DECIMAL(3, 2), nullable=True)
    gaze_direction = Column(String(10), nullable=True)  # 'LEFT', 'RIGHT', 'CENTER'
    ear = Column(DECIMAL(3, 2), nullable=True)  # Eye Aspect Ratio

    # Concentration
    concentration_index = Column(String(20), nullable=True)  # 'ENGAGED', 'DISTRACTED', 'DROWSY'

    # Raw data (for debugging)
    raw_data = Column(JSONB, nullable=True)

    # Relationships
    session = relationship("Session", back_populates="engagement_events")
    student = relationship("Student", back_populates="engagement_events")

    def __repr__(self):
        return f"<EngagementEvent(student_id='{self.student_id}', timestamp='{self.timestamp}')>"


class DistractionLog(Base):
    """Distraction events"""
    __tablename__ = "distraction_logs"

    distraction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.student_id", ondelete="CASCADE"), nullable=False, index=True)
    distraction_type = Column(SQLEnum(DistractionType, name="distraction_type", values_callable=lambda x: [e.value for e in x]), nullable=False, index=True)
    started_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)
    ended_at = Column(TIMESTAMP(timezone=True), nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    video_timestamp = Column(Integer, nullable=True)
    severity = Column(Integer, nullable=True)  # 1-5 scale
    details = Column(JSONB, nullable=True)

    # Relationships
    session = relationship("Session", back_populates="distraction_logs")
    student = relationship("Student", back_populates="distraction_logs")

    def __repr__(self):
        return f"<DistractionLog(type='{self.distraction_type}', student_id='{self.student_id}')>"


class Quiz(Base):
    """Quizzes linked to lectures"""
    __tablename__ = "quizzes"

    quiz_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lecture_id = Column(UUID(as_uuid=True), ForeignKey("lectures.lecture_id", ondelete="CASCADE"), nullable=False, index=True)
    teacher_id = Column(UUID(as_uuid=True), ForeignKey("teachers.teacher_id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    video_timestamp = Column(Integer, nullable=False)  # When to trigger quiz
    time_limit_seconds = Column(Integer, nullable=True)
    passing_score = Column(DECIMAL(5, 2), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    lecture = relationship("Lecture", back_populates="quizzes")
    teacher = relationship("Teacher", back_populates="quizzes")
    questions = relationship("QuizQuestion", back_populates="quiz", cascade="all, delete-orphan")
    attempts = relationship("QuizAttempt", back_populates="quiz", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Quiz(title='{self.title}', lecture_id='{self.lecture_id}')>"


class QuizQuestion(Base):
    """Individual quiz questions"""
    __tablename__ = "quiz_questions"

    question_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    quiz_id = Column(UUID(as_uuid=True), ForeignKey("quizzes.quiz_id", ondelete="CASCADE"), nullable=False, index=True)
    question_text = Column(Text, nullable=False)
    question_type = Column(SQLEnum(QuizQuestionType, name="quiz_question_type", values_callable=lambda x: [e.value for e in x]), nullable=False)
    options = Column(JSONB, nullable=True)  # ["Option A", "Option B", ...]
    correct_answer = Column(Text, nullable=False)
    points = Column(DECIMAL(5, 2), default=1.00)
    explanation = Column(Text, nullable=True)
    order_index = Column(Integer, nullable=False)

    # Relationships
    quiz = relationship("Quiz", back_populates="questions")
    answers = relationship("QuizAnswer", back_populates="question")

    def __repr__(self):
        return f"<QuizQuestion(quiz_id='{self.quiz_id}', order={self.order_index})>"


class QuizAttempt(Base):
    """Student quiz submissions"""
    __tablename__ = "quiz_attempts"

    attempt_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    quiz_id = Column(UUID(as_uuid=True), ForeignKey("quizzes.quiz_id", ondelete="CASCADE"), nullable=False, index=True)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.student_id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id", ondelete="SET NULL"), nullable=True, index=True)
    started_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    submitted_at = Column(TIMESTAMP(timezone=True), nullable=True)
    score = Column(DECIMAL(5, 2), nullable=True)  # Percentage
    total_points = Column(DECIMAL(5, 2), nullable=True)
    earned_points = Column(DECIMAL(5, 2), nullable=True)
    passed = Column(Boolean, nullable=True)
    time_taken_seconds = Column(Integer, nullable=True)

    # Relationships
    quiz = relationship("Quiz", back_populates="attempts")
    student = relationship("Student", back_populates="quiz_attempts")
    session = relationship("Session", back_populates="quiz_attempts")
    answers = relationship("QuizAnswer", back_populates="attempt", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<QuizAttempt(quiz_id='{self.quiz_id}', student_id='{self.student_id}', score={self.score})>"


class QuizAnswer(Base):
    """Individual answers per attempt"""
    __tablename__ = "quiz_answers"

    answer_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    attempt_id = Column(UUID(as_uuid=True), ForeignKey("quiz_attempts.attempt_id", ondelete="CASCADE"), nullable=False, index=True)
    question_id = Column(UUID(as_uuid=True), ForeignKey("quiz_questions.question_id", ondelete="CASCADE"), nullable=False, index=True)
    student_answer = Column(Text, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    points_earned = Column(DECIMAL(5, 2), nullable=True)
    answered_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    attempt = relationship("QuizAttempt", back_populates="answers")
    question = relationship("QuizQuestion", back_populates="answers")

    def __repr__(self):
        return f"<QuizAnswer(attempt_id='{self.attempt_id}', question_id='{self.question_id}')>"


class SessionAnalytics(Base):
    """Aggregated session-level metrics"""
    __tablename__ = "session_analytics"

    analytics_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id", ondelete="CASCADE"), unique=True, nullable=False, index=True)

    # Participation
    total_students = Column(Integer, default=0)
    avg_attendance_duration_seconds = Column(Integer, nullable=True)

    # Engagement
    avg_engagement_score = Column(DECIMAL(5, 2), nullable=True)
    total_distraction_events = Column(Integer, default=0)
    most_common_distraction = Column(SQLEnum(DistractionType, name="distraction_type", values_callable=lambda x: [e.value for e in x]), nullable=True)

    # Emotions
    emotion_distribution = Column(JSONB, nullable=True)
    dominant_emotion = Column(SQLEnum(EmotionType, name="emotion_type", values_callable=lambda x: [e.value for e in x]), nullable=True)

    # Quiz performance
    total_quiz_attempts = Column(Integer, default=0)
    avg_quiz_score = Column(DECIMAL(5, 2), nullable=True)
    quiz_pass_rate = Column(DECIMAL(5, 2), nullable=True)

    # Timestamps
    generated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    session = relationship("Session", back_populates="session_analytics")

    def __repr__(self):
        return f"<SessionAnalytics(session_id='{self.session_id}', avg_engagement={self.avg_engagement_score})>"


class StudentAnalytics(Base):
    """Aggregated student-level metrics per session"""
    __tablename__ = "student_analytics"

    analytics_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.student_id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)

    # Attendance
    duration_seconds = Column(Integer, nullable=True)
    attendance_percentage = Column(DECIMAL(5, 2), nullable=True)

    # Engagement
    engagement_score = Column(DECIMAL(5, 2), nullable=True)
    total_distraction_count = Column(Integer, default=0)
    distraction_breakdown = Column(JSONB, nullable=True)

    # Emotions
    dominant_emotion = Column(SQLEnum(EmotionType, name="emotion_type", values_callable=lambda x: [e.value for e in x]), nullable=True)
    emotion_distribution = Column(JSONB, nullable=True)
    emotion_changes_count = Column(Integer, nullable=True)

    # Focus metrics
    avg_concentration_index = Column(String(20), nullable=True)
    focus_periods = Column(JSONB, nullable=True)

    # Quiz performance
    quiz_score = Column(DECIMAL(5, 2), nullable=True)
    quiz_rank = Column(Integer, nullable=True)

    # Generated data
    generated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    student = relationship("Student", back_populates="student_analytics")
    session = relationship("Session", back_populates="student_analytics")

    def __repr__(self):
        return f"<StudentAnalytics(student_id='{self.student_id}', session_id='{self.session_id}')>"


class AuditLog(Base):
    """Audit logging for admin monitoring"""
    __tablename__ = "audit_logs"

    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(UUID(as_uuid=True), nullable=True)
    ip_address = Column(INET, nullable=True)
    user_agent = Column(String(500), nullable=True)
    details = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog(action='{self.action}', user_id='{self.user_id}')>"


class SystemNotification(Base):
    """System notifications for users"""
    __tablename__ = "system_notifications"

    notification_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=True, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String(50), nullable=True)  # 'info', 'warning', 'error', 'success'
    is_read = Column(Boolean, default=False, index=True)
    link_url = Column(String(500), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)
    read_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="notifications")

    def __repr__(self):
        return f"<SystemNotification(title='{self.title}', user_id='{self.user_id}')>"
