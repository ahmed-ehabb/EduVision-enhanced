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

    Args:
        request: Lecture creation data
        current_user: Authenticated teacher user
        db: Database session

    Returns:
        Created lecture with ID
    """
    # Get teacher profile
    teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
    if not teacher:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Teacher profile not found"
        )

    # Create lecture
    lecture = crud.create_lecture(
        db=db,
        teacher_id=teacher.teacher_id,
        title=request.title,
        description=request.description,
        course_code=request.course_code,
        course_name=request.course_name
    )

    # Log audit event
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


@app.get("/api/lectures", response_model=List[LectureResponse])
def list_lectures(
    current_user: User = Depends(require_role(UserRole.TEACHER)),
    db: Session = Depends(get_db)
):
    """
    List all lectures for the authenticated teacher

    Args:
        current_user: Authenticated teacher user
        db: Database session

    Returns:
        List of lectures
    """
    # Get teacher profile
    teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
    if not teacher:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Teacher profile not found"
        )

    # Get lectures
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


@app.get("/api/lectures/{lecture_id}", response_model=LectureResponse)
def get_lecture(
    lecture_id: str,
    current_user: User = Depends(require_role(UserRole.TEACHER)),
    db: Session = Depends(get_db)
):
    """
    Get a specific lecture by ID

    Args:
        lecture_id: Lecture UUID
        current_user: Authenticated teacher user
        db: Database session

    Returns:
        Lecture details
    """
    lecture = crud.get_lecture_by_id(db, uuid.UUID(lecture_id))
    if not lecture:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lecture not found"
        )

    # Verify ownership
    teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
    if lecture.teacher_id != teacher.teacher_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
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

    Args:
        request: Session creation data
        current_user: Authenticated teacher user
        db: Database session

    Returns:
        Created session with session code
    """
    # Get teacher profile
    teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
    if not teacher:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Teacher profile not found"
        )

    # Verify lecture ownership
    lecture = crud.get_lecture_by_id(db, uuid.UUID(request.lecture_id))
    if not lecture:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lecture not found"
        )

    if lecture.teacher_id != teacher.teacher_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Create session
    session = crud.create_session(
        db=db,
        lecture_id=lecture.lecture_id,
        teacher_id=teacher.teacher_id,
        scheduled_at=request.scheduled_at
    )

    # Log audit event
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


@app.post("/api/sessions/{session_code}/start", response_model=SessionResponse)
def start_session(
    session_code: str,
    current_user: User = Depends(require_role(UserRole.TEACHER)),
    db: Session = Depends(get_db)
):
    """
    Start a session

    Args:
        session_code: Session code (e.g., "ABC123")
        current_user: Authenticated teacher user
        db: Database session

    Returns:
        Updated session
    """
    # Get session
    session = crud.get_session_by_code(db, session_code)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Verify ownership
    teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
    if session.teacher_id != teacher.teacher_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Start session
    session = crud.start_session(db, session.session_id)

    # Log audit event
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

    Args:
        session_code: Session code (e.g., "ABC123")
        current_user: Authenticated teacher user
        db: Database session

    Returns:
        Updated session
    """
    # Get session
    session = crud.get_session_by_code(db, session_code)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Verify ownership
    teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
    if session.teacher_id != teacher.teacher_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # End session
    session = crud.end_session(db, session.session_id)

    # Log audit event
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

    Args:
        session_code: Session code (e.g., "ABC123")
        current_user: Authenticated student user
        db: Database session

    Returns:
        Success message with session info
    """
    # Get session
    session = crud.get_session_by_code(db, session_code)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Check session is active
    if session.status.value != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is {session.status.value}, not active"
        )

    # Get student profile
    student = crud.get_student_by_user_id(db, current_user.user_id)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student profile not found"
        )

    # Create participant record
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

        # Log audit event
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
    events: List[EngagementEventRequest],
    current_user: User = Depends(require_role(UserRole.STUDENT)),
    db: Session = Depends(get_db)
):
    """
    Batch log engagement events for a session

    Args:
        session_id: Session UUID
        events: List of engagement events
        current_user: Authenticated student user
        db: Database session

    Returns:
        Count of events logged
    """
    # Get student profile
    student = crud.get_student_by_user_id(db, current_user.user_id)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student profile not found"
        )

    # Verify session exists
    from models import Session as LectureSession
    session = db.query(LectureSession).filter(
        LectureSession.session_id == uuid.UUID(session_id)
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Prepare events data
    events_data = [
        {
            "session_id": uuid.UUID(session_id),
            "student_id": student.student_id,
            **event.dict()
        }
        for event in events
    ]

    # Batch insert
    count = crud.batch_log_engagement_events(db, events_data)

    return {
        "message": f"Logged {count} engagement events",
        "count": count
    }


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
    Get analytics for a session

    Args:
        session_id: Session UUID
        current_user: Authenticated teacher user
        db: Database session

    Returns:
        Session analytics
    """
    # Verify session exists and ownership
    from models import Session as LectureSession
    session = db.query(LectureSession).filter(
        LectureSession.session_id == uuid.UUID(session_id)
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    teacher = crud.get_teacher_by_user_id(db, current_user.user_id)
    if session.teacher_id != teacher.teacher_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Get analytics
    from models import SessionAnalytics
    analytics = db.query(SessionAnalytics).filter(
        SessionAnalytics.session_id == uuid.UUID(session_id)
    ).first()

    if not analytics:
        # Generate analytics if not exists
        return {
            "session_id": session_id,
            "message": "Analytics not yet generated",
            "total_participants": 0
        }

    return {
        "session_id": str(analytics.session_id),
        "total_participants": analytics.total_participants,
        "avg_engagement_score": float(analytics.avg_engagement_score) if analytics.avg_engagement_score else None,
        "total_distraction_events": analytics.total_distraction_events,
        "most_common_distraction": analytics.most_common_distraction.value if analytics.most_common_distraction else None,
        "emotion_distribution": analytics.emotion_distribution,
        "dominant_emotion": analytics.dominant_emotion.value if analytics.dominant_emotion else None,
        "total_quiz_attempts": analytics.total_quiz_attempts,
        "avg_quiz_score": float(analytics.avg_quiz_score) if analytics.avg_quiz_score else None,
        "generated_at": analytics.generated_at
    }
