"""
Teacher Module V2 REST API
===========================

FastAPI server for the Teacher Module V2 pipeline.
Provides REST endpoints for lecture processing, analysis, and retrieval.

Features:
- Lecture upload and processing
- Async processing with status tracking
- Result retrieval and filtering
- Comprehensive error handling
- Input validation
- CORS support for frontend integration

Author: Ahmed (with AI assistance)
Date: 2025-11-06
"""

import os
import sys
import json
import uuid
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.teacher_module_v2 import TeacherModuleV2
from backend.input_validator import InputValidator
from backend.error_handler import handle_error, ErrorSeverity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== PYDANTIC MODELS ====================

class LectureMetadata(BaseModel):
    """Metadata for a lecture processing request."""
    lecture_title: str = Field(..., min_length=3, max_length=200, description="Title of the lecture")
    teacher_id: Optional[str] = Field(None, description="ID of the teacher")
    subject: Optional[str] = Field("General", description="Subject/topic of the lecture")

    @validator('lecture_title')
    def validate_title(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Lecture title must be at least 3 characters long')
        return v.strip()


class LectureStatus(BaseModel):
    """Status response for a lecture processing job."""
    lecture_id: str = Field(..., description="Unique ID of the lecture")
    status: str = Field(..., description="Status: queued, processing, completed, failed")
    progress: Optional[int] = Field(None, ge=0, le=100, description="Processing progress (0-100%)")
    current_step: Optional[str] = Field(None, description="Current processing step")
    message: Optional[str] = Field(None, description="Status message")
    created_at: str = Field(..., description="ISO timestamp when job was created")
    started_at: Optional[str] = Field(None, description="ISO timestamp when processing started")
    completed_at: Optional[str] = Field(None, description="ISO timestamp when processing completed")
    error: Optional[str] = Field(None, description="Error message if status is 'failed'")


class TranscriptResult(BaseModel):
    """Transcript result."""
    text: str = Field(..., description="Full transcript text")
    segments: List[Dict[str, Any]] = Field(..., description="Transcript segments with timestamps")
    duration: float = Field(..., description="Audio duration in seconds")
    language: str = Field(..., description="Detected language")
    confidence: float = Field(..., ge=0, le=1, description="Average confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")


class EngagementResult(BaseModel):
    """Engagement analysis result."""
    overall_score: float = Field(..., ge=0, le=100, description="Overall engagement score (0-100)")
    overall_label: str = Field(..., description="Overall engagement label (Engaging/Neutral/Boring)")
    segments: List[Dict[str, Any]] = Field(..., description="Per-segment engagement analysis")
    statistics: Dict[str, Any] = Field(..., description="Engagement statistics")


class ContentAlignmentResult(BaseModel):
    """Content alignment result."""
    coverage_score: float = Field(..., ge=0, le=100, description="Overall coverage score (0-100)")
    feedback: str = Field(..., description="Coverage feedback message")
    segments: List[Dict[str, Any]] = Field(..., description="Per-segment alignment analysis")
    statistics: Dict[str, Any] = Field(..., description="Alignment statistics")


class NotesResult(BaseModel):
    """Lecture notes result."""
    markdown: str = Field(..., description="Notes in markdown format")
    bullet_points: List[str] = Field(..., description="Key points as bullet list")
    metadata: Dict[str, Any] = Field(..., description="Notes metadata")


class QuizResult(BaseModel):
    """Quiz generation result."""
    questions: List[Dict[str, Any]] = Field(..., description="List of quiz questions")
    metadata: Dict[str, Any] = Field(..., description="Quiz metadata")


class LectureResult(BaseModel):
    """Complete lecture processing result."""
    lecture_id: str = Field(..., description="Unique ID of the lecture")
    lecture_title: str = Field(..., description="Title of the lecture")
    teacher_id: Optional[str] = Field(None, description="ID of the teacher")
    subject: Optional[str] = Field(None, description="Subject/topic")
    status: str = Field(..., description="Processing status")
    success: bool = Field(..., description="Whether processing succeeded")
    has_warnings: bool = Field(..., description="Whether there are warnings")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    errors: List[str] = Field(default_factory=list, description="List of errors")

    # Processing results
    transcript: Optional[TranscriptResult] = Field(None, description="Transcript result")
    engagement: Optional[EngagementResult] = Field(None, description="Engagement analysis")
    content_alignment: Optional[ContentAlignmentResult] = Field(None, description="Content alignment")
    notes: Optional[NotesResult] = Field(None, description="Lecture notes")
    quiz: Optional[QuizResult] = Field(None, description="Quiz questions")

    # Metadata
    created_at: str = Field(..., description="ISO timestamp when lecture was created")
    processing_time: float = Field(..., description="Total processing time in seconds")
    audio_metadata: Dict[str, Any] = Field(..., description="Audio file metadata")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    lecture_id: Optional[str] = Field(None, description="Lecture ID if applicable")
    timestamp: str = Field(..., description="ISO timestamp of error")


# ==================== FASTAPI APPLICATION ====================

app = FastAPI(
    title="Teacher Module V2 API",
    description="REST API for AI-powered lecture analysis and processing",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "*"  # Allow all for development (restrict in production)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ==================== GLOBAL STATE ====================

# In-memory storage for lecture processing jobs (replace with database in production)
lecture_jobs: Dict[str, Dict[str, Any]] = {}
lecture_results: Dict[str, Dict[str, Any]] = {}

# Teacher Module instance (singleton)
teacher_module: Optional[TeacherModuleV2] = None
input_validator: Optional[InputValidator] = None


# ==================== LIFECYCLE EVENTS ====================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    global teacher_module, input_validator

    try:
        logger.info("=" * 80)
        logger.info("Starting Teacher Module V2 API Server...")
        logger.info("=" * 80)

        # Initialize Teacher Module
        logger.info("Initializing Teacher Module V2...")
        teacher_module = TeacherModuleV2()
        logger.info("Teacher Module V2 initialized successfully")

        # Initialize Input Validator
        logger.info("Initializing Input Validator...")
        input_validator = InputValidator(strict_mode=False)
        logger.info("Input Validator initialized successfully")

        logger.info("=" * 80)
        logger.info("Server started successfully!")
        logger.info("API Documentation: http://localhost:8000/docs")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    try:
        logger.info("Shutting down Teacher Module V2 API Server...")
        # Cleanup can be added here if needed
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# ==================== HEALTH & STATUS ENDPOINTS ====================

@app.get("/", tags=["System"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Teacher Module V2 API",
        "version": "2.0.0",
        "status": "running",
        "description": "AI-powered lecture analysis and processing",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check if teacher module is initialized
        is_healthy = teacher_module is not None and input_validator is not None

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "teacher_module": "ok" if teacher_module else "error",
                "input_validator": "ok" if input_validator else "error",
                "api": "ok"
            },
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.get("/api/status", tags=["System"])
async def get_api_status():
    """Get detailed API status."""
    try:
        return {
            "status": "operational",
            "version": "2.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": {
                "total_jobs": len(lecture_jobs),
                "completed_jobs": len([j for j in lecture_jobs.values() if j["status"] == "completed"]),
                "failed_jobs": len([j for j in lecture_jobs.values() if j["status"] == "failed"]),
                "processing_jobs": len([j for j in lecture_jobs.values() if j["status"] == "processing"])
            },
            "features": {
                "transcription": True,
                "engagement_analysis": True,
                "content_alignment": True,
                "notes_generation": True,
                "quiz_generation": True,
                "translation": False,  # Optional feature
                "input_validation": True,
                "error_handling": True
            }
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# ==================== LECTURE PROCESSING ENDPOINTS ====================

@app.post("/api/lectures/analyze", tags=["Lectures"], response_model=LectureStatus)
async def analyze_lecture(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file of the lecture (WAV, MP3, etc.)"),
    textbook_paragraphs: Optional[str] = Form(None, description="JSON array of textbook paragraphs (optional - defaults to Psychology2e textbook)"),
    lecture_title: str = Form(..., description="Title of the lecture"),
    pdf_path: Optional[str] = Form(None, description="Path to PDF textbook (optional)"),
    teacher_id: Optional[str] = Form(None, description="Teacher ID (optional)"),
    subject: Optional[str] = Form("General", description="Subject/topic (optional)")
):
    """
    Analyze a lecture recording.

    This endpoint accepts an audio file and textbook content, then processes the lecture
    through the complete Teacher Module V2 pipeline:
    1. Input validation
    2. ASR transcription
    3. Engagement analysis
    4. Content alignment
    5. Notes generation
    6. Quiz generation

    Processing is done asynchronously in the background. Use the returned lecture_id
    to check status and retrieve results.

    **Parameters:**
    - audio_file: Audio file (WAV, MP3, M4A, OGG, FLAC, AAC), max 500MB, 30s-2h duration
    - textbook_paragraphs: (Optional) JSON array of textbook paragraphs. If not provided, automatically loads from Psychology2e_WEB.pdf
    - lecture_title: Title of the lecture (3-200 characters)
    - pdf_path: Optional path to PDF textbook
    - teacher_id: Optional teacher identifier
    - subject: Optional subject/topic

    **Returns:**
    - LectureStatus with lecture_id for tracking progress
    """
    try:
        # Generate unique lecture ID
        lecture_id = str(uuid.uuid4())
        logger.info(f"[NEW] Received lecture analysis request: {lecture_id}")
        logger.info(f"      Title: {lecture_title}")
        logger.info(f"      Audio: {audio_file.filename} ({audio_file.content_type})")

        # Parse textbook paragraphs from JSON or load from default PDF
        if textbook_paragraphs:
            # User provided textbook paragraphs
            try:
                textbook_paras = json.loads(textbook_paragraphs)
                if not isinstance(textbook_paras, list):
                    raise ValueError("textbook_paragraphs must be a JSON array")
                logger.info(f"[INFO] Using {len(textbook_paras)} provided textbook paragraphs")
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON format for textbook_paragraphs: {str(e)}"
                )
        else:
            # Load from default Psychology textbook PDF
            logger.info("[INFO] No textbook paragraphs provided - loading from default Psychology2e PDF...")
            default_pdf = Path(__file__).parent.parent / "Psychology2e_WEB.pdf"

            if not default_pdf.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Default textbook PDF not found at {default_pdf}. Please provide textbook_paragraphs."
                )

            try:
                from backend.text_alignment import TextAlignment
                text_aligner = TextAlignment()
                textbook_paras = text_aligner.load_textbook_from_pdf(
                    pdf_path=str(default_pdf),
                    chapter_start_anchor="Psychology is the scientific study of mind and behavior",
                    chapter_end_anchor="Biopsychology is the study of how biology influences behavior"
                )

                if not textbook_paras:
                    raise ValueError("Failed to extract paragraphs from default PDF")

                logger.info(f"[OK] Loaded {len(textbook_paras)} paragraphs from Psychology2e textbook")

            except Exception as e:
                logger.error(f"[ERROR] Failed to load default textbook: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load default textbook: {str(e)}"
                )

        # Save audio file temporarily
        import tempfile
        temp_dir = tempfile.gettempdir()
        audio_ext = Path(audio_file.filename).suffix or ".wav"
        temp_audio_path = os.path.join(temp_dir, f"lecture_{lecture_id}{audio_ext}")

        with open(temp_audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        logger.info(f"[SAVE] Audio saved to: {temp_audio_path}")

        # Quick validation before queuing
        logger.info("[CHECK] Validating inputs...")
        is_valid, errors, metadata = input_validator.validate_pipeline_inputs(
            audio_path=temp_audio_path,
            textbook_paragraphs=textbook_paras,
            pdf_path=pdf_path,
            lecture_title=lecture_title
        )

        if not is_valid:
            # Clean up temp file
            try:
                os.unlink(temp_audio_path)
            except:
                pass

            logger.error(f"[FAIL] Validation failed: {errors}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Input validation failed",
                    "validation_errors": errors,
                    "metadata": metadata
                }
            )

        logger.info("[OK] Validation passed")

        # Create job entry
        lecture_jobs[lecture_id] = {
            "lecture_id": lecture_id,
            "status": "queued",
            "progress": 0,
            "current_step": "Queued for processing",
            "message": "Lecture queued for processing",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "lecture_title": lecture_title,
            "teacher_id": teacher_id,
            "subject": subject,
            "audio_path": temp_audio_path,
            "textbook_paragraphs": textbook_paras,
            "pdf_path": pdf_path,
            "validation_metadata": metadata
        }

        # Add background task for processing
        background_tasks.add_task(
            process_lecture_async,
            lecture_id,
            temp_audio_path,
            textbook_paras,
            lecture_title,
            pdf_path,
            teacher_id,
            subject
        )

        logger.info(f"[QUEUE] Lecture {lecture_id} queued for processing")

        return LectureStatus(
            lecture_id=lecture_id,
            status="queued",
            progress=0,
            current_step="Queued for processing",
            message="Lecture queued successfully. Use /api/lectures/{id} to check status.",
            created_at=lecture_jobs[lecture_id]["created_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Failed to queue lecture: {e}")
        handle_error(e, ErrorSeverity.HIGH, "Lecture Analysis Endpoint")
        raise HTTPException(status_code=500, detail=f"Failed to queue lecture: {str(e)}")


async def process_lecture_async(
    lecture_id: str,
    audio_path: str,
    textbook_paragraphs: List[str],
    lecture_title: str,
    pdf_path: Optional[str],
    teacher_id: Optional[str],
    subject: Optional[str]
):
    """
    Background task to process lecture.

    NOTE: Currently runs synchronously in the event loop because subprocess
    execution causes CUDA initialization issues (segfault during model loading).
    The E2E test works fine, so the issue is specific to subprocess + CUDA.

    TODO: Implement proper async with Celery + Redis for production.
    """
    import asyncio

    try:
        logger.info(f"[START] Processing lecture {lecture_id}")

        # Update status to processing
        lecture_jobs[lecture_id].update({
            "status": "processing",
            "started_at": datetime.utcnow().isoformat(),
            "progress": 10,
            "current_step": "Starting pipeline..."
        })

        # Run pipeline directly (not in subprocess due to CUDA issues)
        logger.info(f"[RUN] Running Teacher Module V2 pipeline...")

        # NOTE: This runs synchronously and will block the event loop
        # For production, use Celery + Redis for proper async execution
        results = teacher_module.process_lecture(
            audio_path=audio_path,
            textbook_paragraphs=textbook_paragraphs,
            lecture_title=lecture_title,
            pdf_path=pdf_path
        )

        logger.info(f"[OK] Pipeline completed successfully")

        # Check if processing succeeded
        if not results.get("success"):
            raise Exception(f"Pipeline failed: {results.get('errors', ['Unknown error'])}")

        logger.info(f"[OK] Pipeline completed successfully")

        # Store results
        lecture_results[lecture_id] = {
            "lecture_id": lecture_id,
            "lecture_title": lecture_title,
            "teacher_id": teacher_id,
            "subject": subject,
            "status": "completed",
            "success": results["success"],
            "has_warnings": results.get("has_warnings", False),
            "warnings": results.get("warnings", []),
            "errors": results.get("errors", []),
            "transcript": results.get("transcript"),
            "engagement": results.get("engagement"),
            "content_alignment": results.get("content_alignment"),
            "notes": results.get("notes"),
            "quiz": results.get("quiz"),
            "created_at": lecture_jobs[lecture_id]["created_at"],
            "processing_time": results.get("total_processing_time", 0),
            "audio_metadata": lecture_jobs[lecture_id]["validation_metadata"]["audio"]
        }

        # Update job status
        lecture_jobs[lecture_id].update({
            "status": "completed",
            "progress": 100,
            "current_step": "Completed",
            "message": "Lecture processed successfully",
            "completed_at": datetime.utcnow().isoformat(),
            "error": None
        })

        logger.info(f"[DONE] Lecture {lecture_id} processing complete")

    except Exception as e:
        import traceback
        logger.error(f"[ERROR] Failed to process lecture {lecture_id}: {e}")
        logger.error(f"[ERROR] Traceback:\n{traceback.format_exc()}")

        # Update job status to failed
        lecture_jobs[lecture_id].update({
            "status": "failed",
            "progress": 0,
            "current_step": "Failed",
            "message": f"Processing failed: {str(e)}",
            "completed_at": datetime.utcnow().isoformat(),
            "error": str(e)
        })

        handle_error(e, ErrorSeverity.HIGH, f"Async Processing (Lecture {lecture_id})")

    finally:
        # Clean up temporary audio file
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                logger.info(f"[CLEAN] Removed temporary file: {audio_path}")
        except Exception as cleanup_error:
            logger.warning(f"[WARN] Failed to clean up temp file: {cleanup_error}")


@app.get("/api/lectures/{lecture_id}", tags=["Lectures"])
async def get_lecture(lecture_id: str):
    """
    Get lecture processing status and results.

    Returns the current status of a lecture processing job. If the job is completed,
    returns the full results including transcript, engagement analysis, content alignment,
    notes, and quiz.

    **Parameters:**
    - lecture_id: Unique lecture identifier returned from /api/lectures/analyze

    **Returns:**
    - LectureStatus if still processing
    - LectureResult if completed
    - Error if failed or not found
    """
    try:
        # Check if lecture exists
        if lecture_id not in lecture_jobs:
            raise HTTPException(
                status_code=404,
                detail=f"Lecture not found: {lecture_id}"
            )

        job = lecture_jobs[lecture_id]

        # If completed, return full results
        if job["status"] == "completed" and lecture_id in lecture_results:
            return lecture_results[lecture_id]

        # If failed, return error
        elif job["status"] == "failed":
            return JSONResponse(
                status_code=500,
                content={
                    "lecture_id": lecture_id,
                    "status": "failed",
                    "error": job.get("error", "Unknown error"),
                    "created_at": job["created_at"],
                    "completed_at": job.get("completed_at")
                }
            )

        # Otherwise, return status
        else:
            return LectureStatus(
                lecture_id=lecture_id,
                status=job["status"],
                progress=job.get("progress", 0),
                current_step=job.get("current_step"),
                message=job.get("message"),
                created_at=job["created_at"],
                started_at=job.get("started_at"),
                completed_at=job.get("completed_at"),
                error=job.get("error")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get lecture {lecture_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve lecture: {str(e)}")


@app.get("/api/lectures", tags=["Lectures"])
async def list_lectures(
    status: Optional[str] = Query(None, description="Filter by status (queued/processing/completed/failed)"),
    teacher_id: Optional[str] = Query(None, description="Filter by teacher ID"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results (1-100)"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List all lectures with optional filtering.

    Returns a paginated list of lectures matching the specified filters.

    **Parameters:**
    - status: Filter by processing status
    - teacher_id: Filter by teacher ID
    - limit: Maximum results per page (1-100)
    - offset: Starting offset for pagination

    **Returns:**
    - List of lecture summaries with metadata
    """
    try:
        # Get all lectures
        all_lectures = []

        for lecture_id, job in lecture_jobs.items():
            # Apply filters
            if status and job["status"] != status:
                continue
            if teacher_id and job.get("teacher_id") != teacher_id:
                continue

            # Build summary
            summary = {
                "lecture_id": lecture_id,
                "lecture_title": job["lecture_title"],
                "teacher_id": job.get("teacher_id"),
                "subject": job.get("subject"),
                "status": job["status"],
                "created_at": job["created_at"],
                "completed_at": job.get("completed_at"),
                "has_results": lecture_id in lecture_results
            }

            all_lectures.append(summary)

        # Sort by created_at (newest first)
        all_lectures.sort(key=lambda x: x["created_at"], reverse=True)

        # Apply pagination
        paginated = all_lectures[offset:offset + limit]

        return {
            "status": "success",
            "total": len(all_lectures),
            "count": len(paginated),
            "offset": offset,
            "limit": limit,
            "lectures": paginated
        }

    except Exception as e:
        logger.error(f"Failed to list lectures: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list lectures: {str(e)}")


@app.delete("/api/lectures/{lecture_id}", tags=["Lectures"])
async def delete_lecture(lecture_id: str):
    """
    Delete a lecture and its results.

    Removes all data associated with a lecture, including job status and results.
    Cannot delete lectures that are currently processing.

    **Parameters:**
    - lecture_id: Unique lecture identifier

    **Returns:**
    - Success message
    """
    try:
        # Check if lecture exists
        if lecture_id not in lecture_jobs:
            raise HTTPException(
                status_code=404,
                detail=f"Lecture not found: {lecture_id}"
            )

        job = lecture_jobs[lecture_id]

        # Cannot delete if still processing
        if job["status"] == "processing":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete lecture while processing. Please wait for completion."
            )

        # Delete from both stores
        del lecture_jobs[lecture_id]
        if lecture_id in lecture_results:
            del lecture_results[lecture_id]

        logger.info(f"[DELETE] Deleted lecture {lecture_id}")

        return {
            "status": "success",
            "message": f"Lecture {lecture_id} deleted successfully",
            "lecture_id": lecture_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete lecture {lecture_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete lecture: {str(e)}")


# ==================== MAIN ====================

if __name__ == "__main__":
    uvicorn.run(
        "teacher_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
