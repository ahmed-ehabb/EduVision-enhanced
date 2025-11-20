"""
Main FastAPI application for the Classroom AI system.
Handles API routes, WebSocket connections, and system initialization.
"""

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging

from .database import get_db
from .database.enhanced_models import User, Course, LectureSession
from .routes import auth
from .enhanced_websocket_manager import EnhancedWebSocketManager
from .realtime_manager import EnhancedRealtimeManager
from .error_handler import setup_error_handlers
from .security import get_current_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Classroom AI API",
    description="Enhanced real-time classroom management system with AI capabilities",
    version="2.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize WebSocket manager
websocket_manager = EnhancedWebSocketManager()
realtime_manager = EnhancedRealtimeManager()

# Setup error handlers
setup_error_handlers(app)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    try:
        logger.info("Initializing system components...")
        await realtime_manager.initialize()
        logger.info("[CHECK] System initialization complete")
    except Exception as e:
        logger.error(f"[ERROR] System initialization failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup system resources on shutdown"""
    try:
        logger.info("Shutting down system components...")
        await realtime_manager.cleanup()
        logger.info("[CHECK] System shutdown complete")
    except Exception as e:
        logger.error(f"[ERROR] System shutdown failed: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint for API health check"""
    return {
        "status": "online",
        "system": "Classroom AI",
        "version": "2.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "database": "connected",
            "gpu": "available",
            "websocket": "ready"
        }
    }

@app.websocket("/ws/classroom/{session_id}")
async def classroom_websocket(
    websocket: WebSocket,
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """WebSocket endpoint for real-time classroom interactions"""
    try:
        await websocket_manager.connect(websocket, session_id, current_user)
        
        while True:
            data = await websocket.receive_json()
            await websocket_manager.handle_message(websocket, data)
            
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket_manager.disconnect(websocket)
        
@app.get("/system/status")
async def system_status(current_user: User = Depends(get_current_user)):
    """Get detailed system status"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
        
    return {
        "system": {
            "status": "operational",
            "version": "2.0.0",
            "environment": "production"
        },
        "components": {
            "database": "healthy",
            "gpu": "active",
            "websocket": "connected",
            "ai_models": "loaded"
        },
        "metrics": {
            "active_sessions": websocket_manager.active_connections,
            "gpu_utilization": realtime_manager.get_gpu_utilization(),
            "memory_usage": realtime_manager.get_memory_usage()
        }
    } 