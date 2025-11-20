"""
CRUD Operations for EduVision Database
Create, Read, Update, Delete operations for all models
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
import uuid

from models import (
    User, Admin, Teacher, Student, FaceRecognitionData,
    RefreshToken, Lecture, Session as LectureSession,
    SessionParticipant, EngagementEvent, DistractionLog,
    Quiz, QuizQuestion, QuizAttempt, QuizAnswer,
    SessionAnalytics, StudentAnalytics, AuditLog, SystemNotification,
    UserRole, SessionStatus, DistractionType, EmotionType
)
from auth import hash_password, verify_password, hash_refresh_token

# ============================================================================
# USER OPERATIONS
# ============================================================================

def create_user(
    db: Session,
    email: str,
    password: str,
    role: UserRole,
    email_verified: bool = False
) -> User:
    """
    Create a new user

    Args:
        db: Database session
        email: User email
        password: Plain text password (will be hashed)
        role: User role (admin/teacher/student)
        email_verified: Email verification status

    Returns:
        Created User object
    """
    hashed_password = hash_password(password)

    user = User(
        email=email,
        password_hash=hashed_password,
        role=role,
        email_verified=email_verified
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: uuid.UUID) -> Optional[User]:
    """Get user by UUID"""
    return db.query(User).filter(User.user_id == user_id).first()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Authenticate user with email and password

    Args:
        db: Database session
        email: User email
        password: Plain text password

    Returns:
        User object if authenticated, None otherwise
    """
    user = get_user_by_email(db, email)

    if not user:
        return None

    if not verify_password(password, user.password_hash):
        return None

    if not user.is_active:
        return None

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    return user


def update_user_password(db: Session, user_id: uuid.UUID, new_password: str) -> bool:
    """
    Update user password

    Args:
        db: Database session
        user_id: User UUID
        new_password: New plain text password

    Returns:
        True if successful, False otherwise
    """
    user = get_user_by_id(db, user_id)

    if not user:
        return False

    user.password_hash = hash_password(new_password)
    db.commit()

    return True


def deactivate_user(db: Session, user_id: uuid.UUID) -> bool:
    """Deactivate user account"""
    user = get_user_by_id(db, user_id)

    if not user:
        return False

    user.is_active = False
    db.commit()

    return True


def activate_user(db: Session, user_id: uuid.UUID) -> bool:
    """Activate user account"""
    user = get_user_by_id(db, user_id)

    if not user:
        return False

    user.is_active = True
    db.commit()

    return True


# ============================================================================
# ADMIN OPERATIONS
# ============================================================================

def create_admin(
    db: Session,
    user_id: uuid.UUID,
    full_name: str,
    phone: Optional[str] = None,
    permissions: Optional[Dict] = None
) -> Admin:
    """Create admin profile"""
    admin = Admin(
        user_id=user_id,
        full_name=full_name,
        phone=phone,
        permissions=permissions or {
            "can_create_users": True,
            "can_delete_users": True,
            "can_view_analytics": True,
            "can_manage_lectures": True
        }
    )

    db.add(admin)
    db.commit()
    db.refresh(admin)

    return admin


def get_admin_by_user_id(db: Session, user_id: uuid.UUID) -> Optional[Admin]:
    """Get admin profile by user ID"""
    return db.query(Admin).filter(Admin.user_id == user_id).first()


# ============================================================================
# TEACHER OPERATIONS
# ============================================================================

def create_teacher(
    db: Session,
    user_id: uuid.UUID,
    full_name: str,
    department: Optional[str] = None,
    bio: Optional[str] = None,
    phone: Optional[str] = None
) -> Teacher:
    """Create teacher profile"""
    teacher = Teacher(
        user_id=user_id,
        full_name=full_name,
        department=department,
        bio=bio,
        phone=phone
    )

    db.add(teacher)
    db.commit()
    db.refresh(teacher)

    return teacher


def get_teacher_by_user_id(db: Session, user_id: uuid.UUID) -> Optional[Teacher]:
    """Get teacher profile by user ID"""
    return db.query(Teacher).filter(Teacher.user_id == user_id).first()


def get_teacher_by_id(db: Session, teacher_id: uuid.UUID) -> Optional[Teacher]:
    """Get teacher by teacher_id"""
    return db.query(Teacher).filter(Teacher.teacher_id == teacher_id).first()


def get_all_teachers(db: Session, skip: int = 0, limit: int = 100) -> List[Teacher]:
    """Get all teachers with pagination"""
    return db.query(Teacher).offset(skip).limit(limit).all()


# ============================================================================
# STUDENT OPERATIONS
# ============================================================================

def create_student(
    db: Session,
    user_id: uuid.UUID,
    full_name: str,
    student_number: Optional[str] = None,
    major: Optional[str] = None,
    year_of_study: Optional[int] = None
) -> Student:
    """Create student profile"""
    student = Student(
        user_id=user_id,
        full_name=full_name,
        student_number=student_number,
        major=major,
        year_of_study=year_of_study
    )

    db.add(student)
    db.commit()
    db.refresh(student)

    return student


def get_student_by_user_id(db: Session, user_id: uuid.UUID) -> Optional[Student]:
    """Get student profile by user ID"""
    return db.query(Student).filter(Student.user_id == user_id).first()


def get_student_by_id(db: Session, student_id: uuid.UUID) -> Optional[Student]:
    """Get student by student_id"""
    return db.query(Student).filter(Student.student_id == student_id).first()


def get_student_by_number(db: Session, student_number: str) -> Optional[Student]:
    """Get student by student number"""
    return db.query(Student).filter(Student.student_number == student_number).first()


def get_all_students(db: Session, skip: int = 0, limit: int = 100) -> List[Student]:
    """Get all students with pagination"""
    return db.query(Student).offset(skip).limit(limit).all()


# ============================================================================
# REFRESH TOKEN OPERATIONS
# ============================================================================

def store_refresh_token(
    db: Session,
    user_id: uuid.UUID,
    token: str,
    expires_at: datetime,
    user_agent: Optional[str] = None,
    ip_address: Optional[str] = None
) -> RefreshToken:
    """
    Store refresh token in database

    Args:
        db: Database session
        user_id: User UUID
        token: Refresh token string
        expires_at: Token expiration datetime
        user_agent: Optional user agent string
        ip_address: Optional IP address

    Returns:
        RefreshToken object
    """
    token_hash = hash_refresh_token(token)

    refresh_token = RefreshToken(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
        user_agent=user_agent,
        ip_address=ip_address
    )

    db.add(refresh_token)
    db.commit()
    db.refresh(refresh_token)

    return refresh_token


def get_refresh_token(db: Session, token: str) -> Optional[RefreshToken]:
    """Get refresh token by token string"""
    token_hash = hash_refresh_token(token)
    return db.query(RefreshToken).filter(
        and_(
            RefreshToken.token_hash == token_hash,
            RefreshToken.revoked == False,
            RefreshToken.expires_at > datetime.utcnow()
        )
    ).first()


def revoke_refresh_token(db: Session, token: str) -> bool:
    """Revoke a refresh token"""
    token_hash = hash_refresh_token(token)
    refresh_token = db.query(RefreshToken).filter(
        RefreshToken.token_hash == token_hash
    ).first()

    if not refresh_token:
        return False

    refresh_token.revoked = True
    refresh_token.revoked_at = datetime.utcnow()
    db.commit()

    return True


def revoke_all_user_tokens(db: Session, user_id: uuid.UUID) -> int:
    """Revoke all refresh tokens for a user (logout from all devices)"""
    count = db.query(RefreshToken).filter(
        and_(
            RefreshToken.user_id == user_id,
            RefreshToken.revoked == False
        )
    ).update({
        "revoked": True,
        "revoked_at": datetime.utcnow()
    })

    db.commit()
    return count


def cleanup_expired_tokens(db: Session) -> int:
    """Delete expired refresh tokens"""
    count = db.query(RefreshToken).filter(
        RefreshToken.expires_at < datetime.utcnow()
    ).delete()

    db.commit()
    return count


# ============================================================================
# LECTURE OPERATIONS
# ============================================================================

def create_lecture(
    db: Session,
    teacher_id: uuid.UUID,
    title: str,
    description: Optional[str] = None,
    video_url: Optional[str] = None,
    textbook_pdf_url: Optional[str] = None,
    **kwargs
) -> Lecture:
    """Create a new lecture"""
    lecture = Lecture(
        teacher_id=teacher_id,
        title=title,
        description=description,
        video_url=video_url,
        textbook_pdf_url=textbook_pdf_url,
        **kwargs
    )

    db.add(lecture)
    db.commit()
    db.refresh(lecture)

    return lecture


def get_lecture_by_id(db: Session, lecture_id: uuid.UUID) -> Optional[Lecture]:
    """Get lecture by ID"""
    return db.query(Lecture).filter(Lecture.lecture_id == lecture_id).first()


def get_lectures_by_teacher(
    db: Session,
    teacher_id: uuid.UUID,
    published_only: bool = False,
    skip: int = 0,
    limit: int = 100
) -> List[Lecture]:
    """Get all lectures by a teacher"""
    query = db.query(Lecture).filter(Lecture.teacher_id == teacher_id)

    if published_only:
        query = query.filter(Lecture.is_published == True)

    return query.order_by(Lecture.created_at.desc()).offset(skip).limit(limit).all()


def publish_lecture(db: Session, lecture_id: uuid.UUID) -> bool:
    """Publish a lecture"""
    lecture = get_lecture_by_id(db, lecture_id)

    if not lecture:
        return False

    lecture.is_published = True
    lecture.published_at = datetime.utcnow()
    db.commit()

    return True


# ============================================================================
# SESSION OPERATIONS
# ============================================================================

def create_session(
    db: Session,
    lecture_id: uuid.UUID,
    teacher_id: uuid.UUID,
    session_code: str,
    scheduled_at: Optional[datetime] = None
) -> LectureSession:
    """Create a new lecture session"""
    session = LectureSession(
        lecture_id=lecture_id,
        teacher_id=teacher_id,
        session_code=session_code,
        scheduled_at=scheduled_at,
        status=SessionStatus.SCHEDULED
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    return session


def get_session_by_code(db: Session, session_code: str) -> Optional[LectureSession]:
    """Get session by session code"""
    return db.query(LectureSession).filter(
        LectureSession.session_code == session_code
    ).first()


def get_session_by_id(db: Session, session_id: uuid.UUID) -> Optional[LectureSession]:
    """Get session by ID"""
    return db.query(LectureSession).filter(
        LectureSession.session_id == session_id
    ).first()


def start_session(db: Session, session_id: uuid.UUID) -> Optional[LectureSession]:
    """Start a session"""
    session = get_session_by_id(db, session_id)

    if not session:
        return None

    session.status = SessionStatus.ACTIVE
    session.started_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)

    return session


def end_session(db: Session, session_id: uuid.UUID) -> Optional[LectureSession]:
    """End a session"""
    session = get_session_by_id(db, session_id)

    if not session:
        return None

    session.status = SessionStatus.COMPLETED
    session.ended_at = datetime.now(timezone.utc)

    if session.started_at:
        duration = (session.ended_at - session.started_at).total_seconds()
        session.duration = int(duration)

    db.commit()
    db.refresh(session)

    return session


# ============================================================================
# SESSION PARTICIPANT OPERATIONS
# ============================================================================

def join_session(
    db: Session,
    session_id: uuid.UUID,
    student_id: uuid.UUID
) -> SessionParticipant:
    """Student joins a session"""
    participant = SessionParticipant(
        session_id=session_id,
        student_id=student_id
    )

    db.add(participant)

    # Update session total_students count
    session = get_session_by_id(db, session_id)
    if session:
        session.total_students = db.query(SessionParticipant).filter(
            SessionParticipant.session_id == session_id
        ).count() + 1

    db.commit()
    db.refresh(participant)

    return participant


def leave_session(
    db: Session,
    session_id: uuid.UUID,
    student_id: uuid.UUID
) -> bool:
    """Student leaves a session"""
    participant = db.query(SessionParticipant).filter(
        and_(
            SessionParticipant.session_id == session_id,
            SessionParticipant.student_id == student_id
        )
    ).first()

    if not participant:
        return False

    participant.left_at = datetime.utcnow()

    if participant.joined_at:
        duration = (participant.left_at - participant.joined_at).total_seconds()
        participant.duration_seconds = int(duration)

    db.commit()

    return True


# ============================================================================
# ENGAGEMENT EVENT OPERATIONS
# ============================================================================

def log_engagement_event(
    db: Session,
    session_id: uuid.UUID,
    student_id: uuid.UUID,
    **metrics
) -> EngagementEvent:
    """
    Log a student engagement event

    Args:
        db: Database session
        session_id: Session UUID
        student_id: Student UUID
        **metrics: Engagement metrics (face_detected, emotion, gaze_ratio, etc.)

    Returns:
        EngagementEvent object
    """
    event = EngagementEvent(
        session_id=session_id,
        student_id=student_id,
        **metrics
    )

    db.add(event)
    db.commit()
    db.refresh(event)

    return event


def batch_log_engagement_events(
    db: Session,
    events_data: List[Dict[str, Any]]
) -> int:
    """
    Batch insert engagement events for performance

    Args:
        db: Database session
        events_data: List of event dictionaries

    Returns:
        Number of events inserted
    """
    events = [EngagementEvent(**data) for data in events_data]
    db.bulk_save_objects(events)
    db.commit()

    return len(events)


# ============================================================================
# DISTRACTION LOG OPERATIONS
# ============================================================================

def log_distraction(
    db: Session,
    session_id: uuid.UUID,
    student_id: uuid.UUID,
    distraction_type: DistractionType,
    severity: int = 3,
    **kwargs
) -> DistractionLog:
    """Log a distraction event"""
    distraction = DistractionLog(
        session_id=session_id,
        student_id=student_id,
        distraction_type=distraction_type,
        severity=severity,
        **kwargs
    )

    db.add(distraction)
    db.commit()
    db.refresh(distraction)

    return distraction


def end_distraction(
    db: Session,
    distraction_id: uuid.UUID
) -> bool:
    """Mark distraction as ended"""
    distraction = db.query(DistractionLog).filter(
        DistractionLog.distraction_id == distraction_id
    ).first()

    if not distraction:
        return False

    distraction.ended_at = datetime.utcnow()

    if distraction.started_at:
        duration = (distraction.ended_at - distraction.started_at).total_seconds()
        distraction.duration_seconds = int(duration)

    db.commit()

    return True


# ============================================================================
# AUDIT LOG OPERATIONS
# ============================================================================

def log_audit_event(
    db: Session,
    user_id: Optional[uuid.UUID],
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[uuid.UUID] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    details: Optional[Dict] = None
) -> AuditLog:
    """Create an audit log entry"""
    log = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip_address,
        user_agent=user_agent,
        details=details
    )

    db.add(log)
    db.commit()
    db.refresh(log)

    return log


# ============================================================================
# NOTIFICATION OPERATIONS
# ============================================================================

def create_notification(
    db: Session,
    user_id: uuid.UUID,
    title: str,
    message: str,
    notification_type: str = "info",
    link_url: Optional[str] = None
) -> SystemNotification:
    """Create a system notification for a user"""
    notification = SystemNotification(
        user_id=user_id,
        title=title,
        message=message,
        type=notification_type,
        link_url=link_url
    )

    db.add(notification)
    db.commit()
    db.refresh(notification)

    return notification


def mark_notification_read(
    db: Session,
    notification_id: uuid.UUID
) -> bool:
    """Mark notification as read"""
    notification = db.query(SystemNotification).filter(
        SystemNotification.notification_id == notification_id
    ).first()

    if not notification:
        return False

    notification.is_read = True
    notification.read_at = datetime.utcnow()
    db.commit()

    return True


def get_user_notifications(
    db: Session,
    user_id: uuid.UUID,
    unread_only: bool = False,
    limit: int = 50
) -> List[SystemNotification]:
    """Get notifications for a user"""
    query = db.query(SystemNotification).filter(
        SystemNotification.user_id == user_id
    )

    if unread_only:
        query = query.filter(SystemNotification.is_read == False)

    return query.order_by(
        SystemNotification.created_at.desc()
    ).limit(limit).all()
