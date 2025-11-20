"""
EduVision Unified API
FastAPI application providing authentication and user management for all modules
"""

from fastapi import FastAPI, Depends, HTTPException, status, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr, Field
import uuid

from database import get_db, check_connection
from models import User, UserRole
import crud
import auth

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="EduVision API",
    description="Unified authentication and data API for Teacher and Student modules",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# CORS Configuration
# ============================================================================

# Allow requests from Teacher Module (port 8000) and Student Module (port 8001)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",  # Teacher Module
        "http://localhost:8001",  # Student Module
        "http://localhost:3000",  # Future frontend
        "http://localhost:5173",  # Vite dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Security Scheme
# ============================================================================

security = HTTPBearer()

# ============================================================================
# Pydantic Schemas
# ============================================================================

class UserRegisterRequest(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str
    role: UserRole

    # Role-specific fields
    department: Optional[str] = None  # For teachers
    student_number: Optional[str] = None  # For students
    major: Optional[str] = None  # For students
    year_of_study: Optional[int] = Field(None, ge=1, le=6)  # For students


class UserLoginRequest(BaseModel):
    """User login request"""
    email: EmailStr
    password: str


class TokenRefreshRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request"""
    current_password: str
    new_password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """User response model"""
    user_id: str
    email: str
    role: str
    is_active: bool
    email_verified: bool
    created_at: datetime
    last_login: Optional[datetime]

    # Profile data
    full_name: Optional[str] = None
    department: Optional[str] = None  # Teacher
    student_number: Optional[str] = None  # Student
    major: Optional[str] = None  # Student

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


# ============================================================================
# LECTURE SCHEMAS
# ============================================================================

class LectureCreateRequest(BaseModel):
    """Request to create a new lecture"""
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    course_code: Optional[str] = Field(None, max_length=50)
    course_name: Optional[str] = Field(None, max_length=255)


class LectureResponse(BaseModel):
    """Lecture response model"""
    lecture_id: str
    teacher_id: str
    title: str
    description: Optional[str]
    course_code: Optional[str]
    course_name: Optional[str]
    is_published: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# SESSION SCHEMAS
# ============================================================================

class SessionCreateRequest(BaseModel):
    """Request to create a new session"""
    lecture_id: str
    scheduled_at: Optional[datetime] = None


class SessionResponse(BaseModel):
    """Session response model"""
    session_id: str
    lecture_id: str
    teacher_id: str
    session_code: str
    status: str
    scheduled_at: Optional[datetime]
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    duration: Optional[int]

    class Config:
        from_attributes = True


# ============================================================================
# ENGAGEMENT SCHEMAS
# ============================================================================

class EngagementEventRequest(BaseModel):
    """Request to log engagement event"""
    face_detected: bool = False
    identity_verified: bool = False
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    head_pitch: Optional[float] = None
    head_yaw: Optional[float] = None
    gaze_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    concentration_index: Optional[str] = None
    video_timestamp: Optional[int] = None


# ============================================================================
# Dependencies
# ============================================================================

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get current authenticated user from JWT token

    Usage:
        @app.get("/protected")
        def protected_route(current_user: User = Depends(get_current_user)):
            return {"user_id": current_user.user_id}
    """
    token = credentials.credentials

    # Decode and verify token
    user_data = auth.get_user_from_token(token)

    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from database
    user = crud.get_user_by_id(db, uuid.UUID(user_data["user_id"]))

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def require_role(required_role: UserRole):
    """
    Dependency factory to require specific user role

    Usage:
        @app.get("/admin-only")
        def admin_route(current_user: User = Depends(require_role(UserRole.ADMIN))):
            return {"message": "Admin access granted"}
    """
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {required_role.value}"
            )
        return current_user

    return role_checker


def get_optional_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Optional authentication - returns None if no token provided

    Usage:
        @app.get("/public-or-private")
        def route(user: Optional[User] = Depends(get_optional_user)):
            if user:
                return {"message": f"Hello {user.email}"}
            return {"message": "Hello guest"}
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization.replace("Bearer ", "")
    user_data = auth.get_user_from_token(token)

    if user_data is None:
        return None

    user = crud.get_user_by_id(db, uuid.UUID(user_data["user_id"]))
    return user if user and user.is_active else None


# ============================================================================
# Helper Functions
# ============================================================================

def create_user_response(user: User, db: Session) -> UserResponse:
    """Create UserResponse from User model"""
    response_data = {
        "user_id": str(user.user_id),
        "email": user.email,
        "role": user.role.value,
        "is_active": user.is_active,
        "email_verified": user.email_verified,
        "created_at": user.created_at,
        "last_login": user.last_login
    }

    # Add role-specific data
    if user.role == UserRole.ADMIN:
        admin = crud.get_admin_by_user_id(db, user.user_id)
        if admin:
            response_data["full_name"] = admin.full_name

    elif user.role == UserRole.TEACHER:
        teacher = crud.get_teacher_by_user_id(db, user.user_id)
        if teacher:
            response_data["full_name"] = teacher.full_name
            response_data["department"] = teacher.department

    elif user.role == UserRole.STUDENT:
        student = crud.get_student_by_user_id(db, user.user_id)
        if student:
            response_data["full_name"] = student.full_name
            response_data["student_number"] = student.student_number
            response_data["major"] = student.major

    return UserResponse(**response_data)


# ============================================================================
# Health Check & Status
# ============================================================================

@app.get("/health")
def health_check():
    """API health check"""
    db_status = check_connection()

    return {
        "status": "healthy" if db_status else "degraded",
        "service": "EduVision Unified API",
        "version": "1.0.0",
        "database": "connected" if db_status else "disconnected",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/")
def root():
    """API root endpoint"""
    return {
        "message": "EduVision Unified API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(
    request: UserRegisterRequest,
    db: Session = Depends(get_db),
    http_request: Request = None
):
    """
    Register a new user (Admin, Teacher, or Student)

    Returns JWT tokens on successful registration
    """
    # Validate email
    if not auth.validate_email(request.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )

    # Check if user already exists
    existing_user = crud.get_user_by_email(db, request.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Validate password strength
    is_valid, error_msg = auth.validate_password_strength(request.password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )

    # Create user
    user = crud.create_user(
        db=db,
        email=request.email,
        password=request.password,
        role=request.role
    )

    # Create role-specific profile
    if request.role == UserRole.ADMIN:
        crud.create_admin(
            db=db,
            user_id=user.user_id,
            full_name=request.full_name
        )

    elif request.role == UserRole.TEACHER:
        crud.create_teacher(
            db=db,
            user_id=user.user_id,
            full_name=request.full_name,
            department=request.department
        )

    elif request.role == UserRole.STUDENT:
        crud.create_student(
            db=db,
            user_id=user.user_id,
            full_name=request.full_name,
            student_number=request.student_number,
            major=request.major,
            year_of_study=request.year_of_study
        )

    # Generate tokens
    tokens = auth.create_token_pair(
        user_id=str(user.user_id),
        email=user.email,
        role=user.role.value
    )

    # Store refresh token
    crud.store_refresh_token(
        db=db,
        user_id=user.user_id,
        token=tokens["refresh_token"],
        expires_at=datetime.utcnow() + timedelta(days=auth.REFRESH_TOKEN_EXPIRE_DAYS),
        ip_address=http_request.client.host if http_request else None,
        user_agent=http_request.headers.get("user-agent") if http_request else None
    )

    # Log audit event
    crud.log_audit_event(
        db=db,
        user_id=user.user_id,
        action="USER_REGISTERED",
        resource_type="user",
        resource_id=user.user_id,
        ip_address=http_request.client.host if http_request else None
    )

    # Create user response
    user_response = create_user_response(user, db)

    return TokenResponse(
        **tokens,
        user=user_response
    )


@app.post("/auth/login", response_model=TokenResponse)
def login(
    request: UserLoginRequest,
    db: Session = Depends(get_db),
    http_request: Request = None
):
    """
    Login with email and password

    Returns JWT tokens on successful authentication
    """
    # Authenticate user
    user = crud.authenticate_user(db, request.email, request.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate tokens
    tokens = auth.create_token_pair(
        user_id=str(user.user_id),
        email=user.email,
        role=user.role.value
    )

    # Store refresh token
    crud.store_refresh_token(
        db=db,
        user_id=user.user_id,
        token=tokens["refresh_token"],
        expires_at=datetime.utcnow() + timedelta(days=auth.REFRESH_TOKEN_EXPIRE_DAYS),
        ip_address=http_request.client.host if http_request else None,
        user_agent=http_request.headers.get("user-agent") if http_request else None
    )

    # Log audit event
    crud.log_audit_event(
        db=db,
        user_id=user.user_id,
        action="USER_LOGIN",
        ip_address=http_request.client.host if http_request else None
    )

    # Create user response
    user_response = create_user_response(user, db)

    return TokenResponse(
        **tokens,
        user=user_response
    )


@app.post("/auth/refresh", response_model=TokenResponse)
def refresh_token(
    request: TokenRefreshRequest,
    db: Session = Depends(get_db)
):
    """
    Refresh access token using refresh token

    Returns new access token and refresh token
    """
    # Verify refresh token
    token_data = auth.verify_token(request.refresh_token, token_type="refresh")

    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if token exists and is not revoked
    db_token = crud.get_refresh_token(db, request.refresh_token)

    if db_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token revoked or not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user
    user = crud.get_user_by_id(db, uuid.UUID(token_data["user_id"]))

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )

    # Revoke old refresh token
    crud.revoke_refresh_token(db, request.refresh_token)

    # Generate new tokens
    tokens = auth.create_token_pair(
        user_id=str(user.user_id),
        email=user.email,
        role=user.role.value
    )

    # Store new refresh token
    crud.store_refresh_token(
        db=db,
        user_id=user.user_id,
        token=tokens["refresh_token"],
        expires_at=datetime.utcnow() + timedelta(days=auth.REFRESH_TOKEN_EXPIRE_DAYS)
    )

    # Create user response
    user_response = create_user_response(user, db)

    return TokenResponse(
        **tokens,
        user=user_response
    )


@app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Logout current user (revoke all refresh tokens)
    """
    # Revoke all user's refresh tokens
    crud.revoke_all_user_tokens(db, current_user.user_id)

    # Log audit event
    crud.log_audit_event(
        db=db,
        user_id=current_user.user_id,
        action="USER_LOGOUT"
    )

    return None


@app.get("/auth/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current authenticated user's information
    """
    return create_user_response(current_user, db)


# ============================================================================
# User Management Endpoints
# ============================================================================

@app.put("/users/me/password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(
    request: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Change current user's password
    """
    # Verify current password
    if not auth.verify_password(request.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )

    # Validate new password
    is_valid, error_msg = auth.validate_password_strength(request.new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )

    # Update password
    success = crud.update_user_password(db, current_user.user_id, request.new_password)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password"
        )

    # Revoke all existing tokens (force re-login)
    crud.revoke_all_user_tokens(db, current_user.user_id)

    # Log audit event
    crud.log_audit_event(
        db=db,
        user_id=current_user.user_id,
        action="PASSWORD_CHANGED"
    )

    return None


# ============================================================================
# LECTURE ENDPOINTS (Teacher Only)
# ============================================================================

@app.post("/api/lectures", response_model=LectureResponse, status_code=status.HTTP_201_CREATED)
def create_lecture(
    request: LectureCreateRequest,
    current_user: User = Depends(require_role(UserRole.TEACHER)),
    db: Session = Depends(get_db)
):
    """
    Create a new lecture (Teacher only)
    """
    teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher profile not found")

    lecture = crud.create_lecture(
        db=db,
        teacher_id=teacher.teacher_id,
        title=request.title,
        description=request.description,
        course_code=request.course_code,
        course_name=request.course_name
    )

    crud.log_audit_event(
        db=db,
        user_id=current_user.user_id,
        action="LECTURE_CREATED",
        resource_type="lecture",
        resource_id=lecture.lecture_id
    )

    return LectureResponse(
        lecture_id=str(lecture.lecture_id),
        teacher_id=str(lecture.teacher_id),
        title=lecture.title,
        description=lecture.description,
        course_code=lecture.course_code,
        course_name=lecture.course_name,
        is_published=lecture.is_published,
        created_at=lecture.created_at,
        updated_at=lecture.updated_at
    )


@app.get("/api/lectures", response_model=list[LectureResponse])
def list_lectures(
    current_user: User = Depends(require_role(UserRole.TEACHER)),
    db: Session = Depends(get_db)
):
    """
    List all lectures for authenticated teacher
    """
    teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher profile not found")

    lectures = crud.get_lectures_by_teacher(db, teacher.teacher_id)

    return [
        LectureResponse(
            lecture_id=str(l.lecture_id),
            teacher_id=str(l.teacher_id),
            title=l.title,
            description=l.description,
            course_code=l.course_code,
            course_name=l.course_name,
            is_published=l.is_published,
            created_at=l.created_at,
            updated_at=l.updated_at
        )
        for l in lectures
    ]


# ============================================================================
# SESSION ENDPOINTS (Teacher Only)
# ============================================================================

@app.post("/api/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    request: SessionCreateRequest,
    current_user: User = Depends(require_role(UserRole.TEACHER)),
    db: Session = Depends(get_db)
):
    """
    Create a new session for a lecture
    """
    teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher profile not found")

    lecture = crud.get_lecture_by_id(db, uuid.UUID(request.lecture_id))
    if not lecture or lecture.teacher_id != teacher.teacher_id:
        raise HTTPException(status_code=404, detail="Lecture not found")

    # Generate unique session code
    session_code = auth.generate_session_code()

    session = crud.create_session(
        db=db,
        lecture_id=lecture.lecture_id,
        teacher_id=teacher.teacher_id,
        session_code=session_code,
        scheduled_at=request.scheduled_at
    )

    crud.log_audit_event(
        db=db,
        user_id=current_user.user_id,
        action="SESSION_CREATED",
        resource_type="session",
        resource_id=session.session_id
    )

    return SessionResponse(
        session_id=str(session.session_id),
        lecture_id=str(session.lecture_id),
        teacher_id=str(session.teacher_id),
        session_code=session.session_code,
        status=session.status.value,
        scheduled_at=session.scheduled_at,
        started_at=session.started_at,
        ended_at=session.ended_at,
        duration=session.duration
    )


@app.get("/api/sessions", response_model=list[SessionResponse])
def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all sessions for the current user (teacher's sessions or student's joined sessions)
    """
    if current_user.role == UserRole.TEACHER:
        teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
        if not teacher:
            return []

        from models import LectureSession
        sessions = db.query(LectureSession).filter(
            LectureSession.teacher_id == teacher.teacher_id
        ).order_by(LectureSession.created_at.desc()).all()

    elif current_user.role == UserRole.STUDENT:
        student = crud.get_student_by_user_id(db, current_user.user_id)
        if not student:
            return []

        from models import LectureSession, SessionParticipant
        sessions = db.query(LectureSession).join(
            SessionParticipant,
            SessionParticipant.session_id == LectureSession.session_id
        ).filter(
            SessionParticipant.student_id == student.student_id
        ).order_by(LectureSession.created_at.desc()).all()
    else:
        return []

    return [
        SessionResponse(
            session_id=str(s.session_id),
            lecture_id=str(s.lecture_id),
            teacher_id=str(s.teacher_id),
            session_code=s.session_code,
            status=s.status.value,
            scheduled_at=s.scheduled_at,
            started_at=s.started_at,
            ended_at=s.ended_at,
            duration=s.duration
        )
        for s in sessions
    ]


@app.post("/api/sessions/{session_code}/start", response_model=SessionResponse)
def start_session(
    session_code: str,
    current_user: User = Depends(require_role(UserRole.TEACHER)),
    db: Session = Depends(get_db)
):
    """
    Start a session
    """
    session = crud.get_session_by_code(db, session_code)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
    if session.teacher_id != teacher.teacher_id:
        raise HTTPException(status_code=403, detail="Access denied")

    session = crud.start_session(db, session.session_id)
    if not session:
        raise HTTPException(status_code=500, detail="Failed to start session")

    crud.log_audit_event(
        db=db,
        user_id=current_user.user_id,
        action="SESSION_STARTED",
        resource_type="session",
        resource_id=session.session_id
    )

    return SessionResponse(
        session_id=str(session.session_id),
        lecture_id=str(session.lecture_id),
        teacher_id=str(session.teacher_id),
        session_code=session.session_code,
        status=session.status.value,
        scheduled_at=session.scheduled_at,
        started_at=session.started_at,
        ended_at=session.ended_at,
        duration=session.duration
    )


@app.post("/api/sessions/{session_code}/end", response_model=SessionResponse)
def end_session(
    session_code: str,
    current_user: User = Depends(require_role(UserRole.TEACHER)),
    db: Session = Depends(get_db)
):
    """
    End a session
    """
    session = crud.get_session_by_code(db, session_code)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
    if session.teacher_id != teacher.teacher_id:
        raise HTTPException(status_code=403, detail="Access denied")

    session = crud.end_session(db, session.session_id)
    if not session:
        raise HTTPException(status_code=500, detail="Failed to end session")

    crud.log_audit_event(
        db=db,
        user_id=current_user.user_id,
        action="SESSION_ENDED",
        resource_type="session",
        resource_id=session.session_id
    )

    return SessionResponse(
        session_id=str(session.session_id),
        lecture_id=str(session.lecture_id),
        teacher_id=str(session.teacher_id),
        session_code=session.session_code,
        status=session.status.value,
        scheduled_at=session.scheduled_at,
        started_at=session.started_at,
        ended_at=session.ended_at,
        duration=session.duration
    )


# ============================================================================
# ENGAGEMENT ENDPOINTS (Student Only)
# ============================================================================

@app.post("/api/sessions/{session_code}/join")
def join_session(
    session_code: str,
    current_user: User = Depends(require_role(UserRole.STUDENT)),
    db: Session = Depends(get_db)
):
    """
    Student joins a session
    """
    session = crud.get_session_by_code(db, session_code)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status.value != "active":
        raise HTTPException(status_code=400, detail=f"Session is {session.status.value}")

    student = crud.get_student_by_user_id(db, current_user.user_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student profile not found")

    from models import SessionParticipant
    participant = db.query(SessionParticipant).filter(
        SessionParticipant.session_id == session.session_id,
        SessionParticipant.student_id == student.student_id
    ).first()

    if not participant:
        participant = SessionParticipant(
            session_id=session.session_id,
            student_id=student.student_id
        )
        db.add(participant)
        db.commit()

        crud.log_audit_event(
            db=db,
            user_id=current_user.user_id,
            action="SESSION_JOINED",
            resource_type="session",
            resource_id=session.session_id
        )

    return {
        "message": "Joined session successfully",
        "session_code": session.session_code,
        "session_id": str(session.session_id),
        "lecture_title": session.lecture.title
    }


@app.post("/api/sessions/{session_id}/engagement", status_code=status.HTTP_201_CREATED)
def log_engagement(
    session_id: str,
    events: list[EngagementEventRequest],
    current_user: User = Depends(require_role(UserRole.STUDENT)),
    db: Session = Depends(get_db)
):
    """
    Batch log engagement events
    """
    student = crud.get_student_by_user_id(db, current_user.user_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student profile not found")

    from models import Session as LectureSession
    session = db.query(LectureSession).filter(
        LectureSession.session_id == uuid.UUID(session_id)
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    events_data = [
        {
            "session_id": uuid.UUID(session_id),
            "student_id": student.student_id,
            **event.dict()
        }
        for event in events
    ]

    count = crud.batch_log_engagement_events(db, events_data)

    return {"message": f"Logged {count} engagement events", "count": count}


# ============================================================================
# ANALYTICS ENDPOINTS (Teacher Only)
# ============================================================================

@app.get("/api/sessions/{session_id}/analytics")
def get_session_analytics(
    session_id: str,
    current_user: User = Depends(require_role(UserRole.TEACHER)),
    db: Session = Depends(get_db)
):
    """
    Get session analytics
    """
    from models import Session as LectureSession, SessionAnalytics
    session = db.query(LectureSession).filter(
        LectureSession.session_id == uuid.UUID(session_id)
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
    if session.teacher_id != teacher.teacher_id:
        raise HTTPException(status_code=403, detail="Access denied")

    analytics = db.query(SessionAnalytics).filter(
        SessionAnalytics.session_id == uuid.UUID(session_id)
    ).first()

    if not analytics:
        return {"session_id": session_id, "message": "Analytics not yet generated"}

    return {
        "session_id": str(analytics.session_id),
        "total_participants": analytics.total_participants,
        "avg_engagement_score": float(analytics.avg_engagement_score) if analytics.avg_engagement_score else None,
        "total_distraction_events": analytics.total_distraction_events,
        "most_common_distraction": analytics.most_common_distraction.value if analytics.most_common_distraction else None,
        "emotion_distribution": analytics.emotion_distribution,
        "dominant_emotion": analytics.dominant_emotion.value if analytics.dominant_emotion else None,
        "generated_at": analytics.generated_at
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("Starting EduVision Unified API")
    print("=" * 70)
    print("API Documentation: http://localhost:8002/docs")
    print("Health Check: http://localhost:8002/health")
    print("=" * 70)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
