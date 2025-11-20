"""
WebSocket Manager for Real-Time Communication
This module provides managers for two types of WebSocket communications:
1.  `ConnectionManager`: For broadcasting JSON-based messages (like status,
    analytics, and alerts) from the server to all connected clients.
2.  `VideoWebSocketManager`: For receiving binary video streams from individual
    student clients to the server for AI processing.

Author: Ahmed
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import numpy as np
import cv2
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# --- Manager for JSON-based Broadcasting ---


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_data: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_data[websocket] = {
            "client_id": client_id,
            "connected_at": asyncio.get_event_loop().time(),
        }
        logger.info(f"WebSocket connected: {client_id or 'anonymous'}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_data = self.connection_data.pop(websocket, {})
            client_id = client_data.get("client_id", "anonymous")
            logger.info(f"WebSocket disconnected: {client_id}")

    async def broadcast(self, message: Dict[str, Any]):
        disconnected = []
        for connection in self.active_connections:
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_json(message)
                else:
                    disconnected.append(connection)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        for connection in disconnected:
            self.disconnect(connection)


# Global instance for JSON broadcasting
connection_manager = ConnectionManager()


# --- Manager for Video Streaming ---


class VideoWebSocketManager:
    """
    Manages WebSocket connections and video frame buffers for all students.
    """

    def __init__(self):
        self.active_connections: Dict[str, Any] = {}
        self.frame_buffers: Dict[str, asyncio.Queue] = {}
        print("[OK] Video WebSocket Manager initialized.")

    async def register_student(self, websocket: Any, student_id: str):
        self.active_connections[student_id] = websocket
        self.frame_buffers[student_id] = asyncio.Queue(maxsize=10)
        print(f"[OK] Student '{student_id}' registered for video stream.")

    async def unregister_student(self, student_id: str):
        if student_id in self.active_connections:
            del self.active_connections[student_id]
        if student_id in self.frame_buffers:
            del self.frame_buffers[student_id]
        print(f"[OK] Student '{student_id}' unregistered.")

    async def handle_video_connection(self, websocket: Any, student_id: str):
        """
        Handles an incoming video WebSocket connection and routes frames.
        """
        await self.register_student(websocket, student_id)
        try:
            async for message in websocket:
                if student_id in self.frame_buffers:
                    if self.frame_buffers[student_id].full():
                        await self.frame_buffers[student_id].get()

                    nparr = np.frombuffer(message, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        await self.frame_buffers[student_id].put(frame)
        except WebSocketDisconnect:
            logger.info(f"Video WebSocket for student '{student_id}' disconnected.")
        except Exception as e:
            logger.error(f"[ERROR] Video WebSocket error for {student_id}: {e}")
        finally:
            await self.unregister_student(student_id)

    def get_video_queue(self, student_id: str) -> Optional[asyncio.Queue]:
        """
        Returns the video frame queue for a given student.
        """
        return self.frame_buffers.get(student_id)


# Singleton instance for video streaming
video_ws_manager = VideoWebSocketManager()


# --- Manager for Live Classroom Sessions ---

@dataclass
class SessionParticipant:
    websocket: WebSocket
    role: str  # "teacher" or "student"
    user_id: str
    name: str
    connected_at: datetime

@dataclass
class LiveSession:
    session_id: str
    teacher_id: str
    course_id: str
    participants: Dict[str, SessionParticipant]  # user_id -> participant
    started_at: datetime
    is_active: bool = True

class WebSocketManager:
    def __init__(self):
        self.active_sessions: Dict[str, LiveSession] = {}  # session_id -> LiveSession
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.logger = logging.getLogger(__name__)

    async def connect_participant(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: str,
        role: str,
        name: str
    ) -> bool:
        """Connect a new participant to a session."""
        try:
            await websocket.accept()
            
            # Create session if it doesn't exist (teacher only)
            if session_id not in self.active_sessions:
                if role != "teacher":
                    await websocket.send_json({
                        "type": "error",
                        "message": "Session not found"
                    })
                    return False
                    
                self.active_sessions[session_id] = LiveSession(
                    session_id=session_id,
                    teacher_id=user_id,
                    course_id="",  # Will be set when session starts
                    participants={},
                    started_at=datetime.now()
                )

            session = self.active_sessions[session_id]
            
            # Add participant to session
            session.participants[user_id] = SessionParticipant(
                websocket=websocket,
                role=role,
                user_id=user_id,
                name=name,
                connected_at=datetime.now()
            )
            
            # Map user to session
            self.user_sessions[user_id] = session_id
            
            # Notify others about new participant
            await self.broadcast_to_session(
                session_id,
                {
                    "type": "participant_joined",
                    "user_id": user_id,
                    "name": name,
                    "role": role
                },
                exclude_user=user_id
            )
            
            # Send current session state to new participant
            await websocket.send_json({
                "type": "session_state",
                "session_id": session_id,
                "participants": [
                    {
                        "user_id": p.user_id,
                        "name": p.name,
                        "role": p.role
                    }
                    for p in session.participants.values()
                ]
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting participant: {e}")
            return False

    async def disconnect_participant(self, user_id: str):
        """Disconnect a participant from their session."""
        try:
            if user_id not in self.user_sessions:
                return
                
            session_id = self.user_sessions[user_id]
            session = self.active_sessions[session_id]
            
            if user_id in session.participants:
                participant = session.participants[user_id]
                
                # Remove from mappings
                del session.participants[user_id]
                del self.user_sessions[user_id]
                
                # Notify others
                await self.broadcast_to_session(
                    session_id,
                    {
                        "type": "participant_left",
                        "user_id": user_id,
                        "name": participant.name
                    }
                )
                
                # Close websocket
                await participant.websocket.close()
                
                # Clean up empty sessions
                if not session.participants:
                    del self.active_sessions[session_id]
                
        except Exception as e:
            self.logger.error(f"Error disconnecting participant: {e}")

    async def broadcast_to_session(
        self,
        session_id: str,
        message: dict,
        exclude_user: Optional[str] = None
    ):
        """Broadcast a message to all participants in a session."""
        if session_id not in self.active_sessions:
            return
            
        session = self.active_sessions[session_id]
        
        # Convert message to JSON string
        message_str = json.dumps(message)
        
        # Send to all participants except excluded user
        for user_id, participant in session.participants.items():
            if user_id != exclude_user:
                try:
                    await participant.websocket.send_text(message_str)
                except Exception as e:
                    self.logger.error(f"Error sending to {user_id}: {e}")
                    await self.disconnect_participant(user_id)

    async def send_to_user(self, user_id: str, message: dict) -> bool:
        """Send a message to a specific user."""
        try:
            if user_id not in self.user_sessions:
                return False
                
            session_id = self.user_sessions[user_id]
            session = self.active_sessions[session_id]
            
            if user_id not in session.participants:
                return False
                
            participant = session.participants[user_id]
            await participant.websocket.send_json(message)
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending to user {user_id}: {e}")
            return False

    async def start_session(self, session_id: str, course_id: str) -> bool:
        """Start a live session."""
        if session_id not in self.active_sessions:
            return False
            
        session = self.active_sessions[session_id]
        session.course_id = course_id
        
        await self.broadcast_to_session(
            session_id,
            {
                "type": "session_started",
                "session_id": session_id,
                "course_id": course_id,
                "started_at": session.started_at.isoformat()
            }
        )
        
        return True

    async def end_session(self, session_id: str):
        """End a live session."""
        if session_id not in self.active_sessions:
            return
            
        session = self.active_sessions[session_id]
        
        # Notify all participants
        await self.broadcast_to_session(
            session_id,
            {
                "type": "session_ended",
                "session_id": session_id
            }
        )
        
        # Disconnect all participants
        for user_id in list(session.participants.keys()):
            await self.disconnect_participant(user_id)
            
        # Remove session
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

    async def update_engagement(
        self,
        session_id: str,
        student_id: str,
        engagement_data: dict
    ):
        """Update student engagement data."""
        if session_id not in self.active_sessions:
            return
            
        session = self.active_sessions[session_id]
        
        # Find teacher
        teacher_id = session.teacher_id
        
        # Send to teacher only
        await self.send_to_user(
            teacher_id,
            {
                "type": "engagement_update",
                "student_id": student_id,
                "data": engagement_data
            }
        )

    async def update_attendance(
        self,
        session_id: str,
        attendance_data: dict
    ):
        """Update attendance data for the session."""
        if session_id not in self.active_sessions:
            return
            
        # Broadcast to all participants
        await self.broadcast_to_session(
            session_id,
            {
                "type": "attendance_update",
                "data": attendance_data
            }
        )

    def get_session_participants(self, session_id: str) -> List[dict]:
        """Get list of participants in a session."""
        if session_id not in self.active_sessions:
            return []
            
        session = self.active_sessions[session_id]
        return [
            {
                "user_id": p.user_id,
                "name": p.name,
                "role": p.role,
                "connected_at": p.connected_at.isoformat()
            }
            for p in session.participants.values()
        ]

    def get_user_session(self, user_id: str) -> Optional[str]:
        """Get the session ID for a user."""
        return self.user_sessions.get(user_id)

    def cleanup(self):
        """Clean up all sessions."""
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            asyncio.create_task(self.end_session(session_id))

# Global instance
websocket_manager = WebSocketManager()
