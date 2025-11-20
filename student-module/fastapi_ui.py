"""
FastAPI Web Interface for Distraction Detection System
Quick UI to test and demonstrate features for graduation project
"""

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import json
import asyncio
import threading
import subprocess
import time
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import uvicorn
from datetime import datetime, timedelta
import io

# Import our analysis modules
try:
    from util.analysis_realtime_cv import analysis
    from util.identity_verifier import IDVerifier
    import util.config as config_module
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import analysis modules: {e}")
    ANALYSIS_AVAILABLE = False
    config_module = None

# Initialize FastAPI
app = FastAPI(title="Distraction Detection System", description="Real-time student monitoring dashboard")

# Global variables
detection_process = None
is_running = False
connected_clients = []
camera_active = False
video_capture = None
analysis_cv = None
identity_verifier = None

# Configuration constants
NO_FACE_SECS = config_module.NO_FACE_SECS if ANALYSIS_AVAILABLE else 5
OFF_SCREEN_SECS = config_module.OFF_SCREEN_SECS if ANALYSIS_AVAILABLE else 2

# Ensure directories exist
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("frontend_outputs/student").mkdir(parents=True, exist_ok=True)
Path("frontend_outputs/teacher").mkdir(parents=True, exist_ok=True)

# Templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    return str(timedelta(seconds=int(seconds)))

def initialize_camera_and_analysis():
    """Initialize camera and analysis components"""
    global video_capture, analysis_cv, identity_verifier
    
    try:
        # Initialize camera
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            return False, "Could not open camera"
        
        # Set camera properties for better performance
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        
        if ANALYSIS_AVAILABLE:
            # Initialize identity verifier first
            identity_verifier = IDVerifier()
            # Pass identity verifier to analysis
            analysis_cv = analysis(identity_verifier=identity_verifier)
            print(f"[INIT] Analysis initialized with identity verifier")

        return True, "Camera and analysis initialized successfully"
    except Exception as e:
        return False, f"Failed to initialize camera/analysis: {str(e)}"

def cleanup_camera():
    """Cleanup camera resources"""
    global video_capture, camera_active
    
    camera_active = False
    if video_capture:
        video_capture.release()
        video_capture = None

def generate_frames():
    """Generate video frames for streaming"""
    global video_capture, camera_active, analysis_cv
    
    while camera_active and video_capture:
        try:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Process frame with analysis if available
            processed_frame = frame.copy()
            
            if ANALYSIS_AVAILABLE and analysis_cv:
                try:
                    # Detect face and add overlays
                    analysis_frame = analysis_cv.detect_face(frame)
                    if analysis_frame is not None:
                        processed_frame = analysis_frame
                    
                    # Add status text overlay
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    color = (0, 255, 0)  # Green color
                    bg_color = (0, 0, 0)  # Black background
                    
                    # Get current status from analysis
                    status = analysis_cv.gen_concentration_index()
                    status_text = f"Status: {status}"
                    timestamp_text = f"Time: {datetime.now().strftime('%H:%M:%S')}"
                    
                    # Get text size
                    (text_width, text_height), _ = cv2.getTextSize(status_text, font, font_scale, thickness)
                    
                    # Calculate text position
                    text_x = 10
                    text_y = processed_frame.shape[0] - 50  # 50 pixels from bottom
                    
                    # Draw background rectangle for status
                    cv2.rectangle(processed_frame, 
                                (text_x - 5, text_y - text_height - 5),
                                (text_x + text_width + 5, text_y + 5),
                                bg_color, -1)
                    
                    # Draw status text
                    cv2.putText(processed_frame, status_text,
                               (text_x, text_y), font, font_scale, color, thickness)
                    
                    # Draw timestamp
                    cv2.putText(processed_frame, timestamp_text,
                               (text_x, text_y + 25), font, 0.5, (255, 255, 255), 1)
                    
                except Exception as e:
                    print(f"Error in frame analysis: {e}")
            
            # Add basic info overlay if analysis not available
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(processed_frame, "Camera Active - Analysis Module Not Available",
                           (10, 30), font, 0.6, (0, 255, 255), 2)
                cv2.putText(processed_frame, f"Time: {datetime.now().strftime('%H:%M:%S')}",
                           (10, 60), font, 0.5, (255, 255, 255), 1)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to control frame rate
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Error generating frame: {e}")
            break

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and Docker"""
    import cv2

    health_status = {
        "status": "healthy",
        "service": "Student Distraction Detection Module",
        "version": "1.0.0",
        "port": 8001,
        "components": {
            "api": "operational",
            "opencv": "installed" if cv2.__version__ else "missing",
            "face_recognition": "ready" if hasattr(cv2, 'face') else "missing",
            "camera": "active" if camera_active else "inactive",
            "analysis": "running" if is_running else "stopped"
        },
        "models": {
            "emotion_detection": "loaded" if ANALYSIS_AVAILABLE else "unavailable",
            "face_recognition": "loaded" if identity_verifier else "not_initialized",
            "pose_estimation": "loaded" if ANALYSIS_AVAILABLE else "unavailable"
        }
    }

    # Check if critical components are missing
    if not hasattr(cv2, 'face'):
        health_status["status"] = "degraded"
        health_status["components"]["face_recognition"] = "error: opencv-contrib-python required"

    return health_status


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    """Video streaming route"""
    global camera_active
    
    if not camera_active:
        # Try to initialize camera
        success, message = initialize_camera_and_analysis()
        if not success:
            raise HTTPException(status_code=500, detail=message)
        camera_active = True
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/api/start-camera")
async def start_camera():
    """Start camera feed"""
    global camera_active
    
    if camera_active:
        return {"message": "Camera is already active", "status": "running"}
    
    success, message = initialize_camera_and_analysis()
    if success:
        camera_active = True
        return {"message": message, "status": "started"}
    else:
        return {"message": message, "status": "error"}

@app.post("/api/stop-camera")
async def stop_camera():
    """Stop camera feed"""
    global camera_active
    
    if not camera_active:
        return {"message": "Camera is not active", "status": "stopped"}
    
    cleanup_camera()
    return {"message": "Camera stopped successfully", "status": "stopped"}

@app.get("/api/camera-status")
async def get_camera_status():
    """Get camera status"""
    global camera_active, video_capture
    
    return {
        "active": camera_active,
        "available": video_capture is not None,
        "analysis_available": ANALYSIS_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/status")
async def get_system_status():
    """Get current system status"""
    global is_running, camera_active
    return {
        "running": is_running,
        "camera_active": camera_active,
        "analysis_available": ANALYSIS_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "outputs_available": len(list(Path("frontend_outputs/student").glob("current_*.json"))) > 0
    }

@app.post("/api/start")
async def start_detection(background_tasks: BackgroundTasks):
    """Start the distraction detection system"""
    global detection_process, is_running
    
    if is_running:
        return {"message": "System is already running", "status": "running"}
    
    try:
        # Check if camera is available first
        import cv2
        test_cap = cv2.VideoCapture(0)
        if not test_cap.isOpened():
            test_cap.release()
            return {
                "message": "Camera not available! Please check camera permissions and connections.",
                "status": "error"
            }
        test_cap.release()
        
        # Start the headless detection process
        detection_process = subprocess.Popen(
            ["python", "run_headless_detection.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            cwd=".",
            text=True  # Handle text output properly
        )
        is_running = True
        
        # Monitor the process
        background_tasks.add_task(monitor_detection_process)
        
        return {
            "message": "Detection system started successfully! Running in headless mode.",
            "status": "started"
        }
    except Exception as e:
        return {"message": f"Failed to start detection system: {str(e)}", "status": "error"}

@app.post("/api/stop")
async def stop_detection():
    """Stop the distraction detection system"""
    global detection_process, is_running
    
    if not is_running:
        return {"message": "System is not running", "status": "stopped"}
    
    try:
        if detection_process:
            detection_process.terminate()
            detection_process.wait(timeout=5)
        is_running = False
        
        # Also stop camera when stopping detection
        cleanup_camera()
        
        return {"message": "Detection system and camera stopped successfully", "status": "stopped"}
    except Exception as e:
        return {"message": f"Failed to stop detection system: {str(e)}", "status": "error"}

@app.get("/api/student-data/{student_name}")
async def get_student_data(student_name: str):
    """Get current student data"""
    try:
        file_path = Path(f"frontend_outputs/student/current_{student_name}.json")
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        else:
            raise HTTPException(status_code=404, detail="Student data not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/teacher-data/{student_name}")
async def get_teacher_data(student_name: str):
    """Get current teacher dashboard data"""
    try:
        file_path = Path(f"frontend_outputs/teacher/current_{student_name}.json")
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        else:
            raise HTTPException(status_code=404, detail="Teacher data not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/students")
async def get_available_students():
    """Get list of available students"""
    try:
        student_files = list(Path("frontend_outputs/student").glob("current_*.json"))
        students = []
        for file in student_files:
            name = file.name.replace("current_", "").replace(".json", "")
            students.append(name)
        return {"students": students}
    except Exception as e:
        return {"students": [], "error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        while True:
            # Send periodic updates
            students = await get_available_students()
            if students["students"]:
                for student in students["students"]:
                    try:
                        student_data = await get_student_data(student)
                        await websocket.send_json({
                            "type": "student_update",
                            "student": student,
                            "data": student_data
                        })
                    except:
                        pass
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global detection_process, is_running
    
    print("Shutting down application...")
    
    # Stop detection process
    if detection_process and is_running:
        try:
            detection_process.terminate()
            detection_process.wait(timeout=5)
        except Exception as e:
            print(f"Error stopping detection process: {e}")
    
    # Cleanup camera
    cleanup_camera()
    
    print("Application shutdown complete.")

async def monitor_detection_process():
    """Monitor the detection process"""
    global detection_process, is_running
    
    if detection_process:
        try:
            # Wait for process to complete
            return_code = detection_process.wait()
            print(f"Detection process ended with return code: {return_code}")
            
            # Read any remaining output
            if detection_process.stdout:
                output = detection_process.stdout.read().decode()
                if output:
                    print(f"Process output: {output}")
                    
        except Exception as e:
            print(f"Error monitoring detection process: {e}")
        finally:
            is_running = False

# Test endpoints
@app.get("/api/test-camera")
async def test_camera():
    """Test if camera is working"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return {
                "camera_available": False,
                "message": "Camera not accessible. Check permissions and connections.",
                "status": "error"
            }
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return {
                "camera_available": False,
                "message": "Camera accessible but cannot read frames.",
                "status": "error"
            }
        
        return {
            "camera_available": True,
            "message": "Camera is working correctly!",
            "frame_size": f"{frame.shape[1]}x{frame.shape[0]}",
            "status": "success"
        }
        
    except Exception as e:
        return {
            "camera_available": False,
            "message": f"Camera test failed: {str(e)}",
            "status": "error"
        }

@app.post("/api/test-outputs")
async def test_outputs():
    """Generate test outputs without camera"""
    try:
        # Run the test script
        result = subprocess.run(
            ["python", "test_enhanced_outputs.py"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if result.returncode == 0:
            return {
                "message": "Test outputs generated successfully",
                "status": "success",
                "output": result.stdout
            }
        else:
            return {
                "message": "Test failed",
                "status": "error",
                "error": result.stderr
            }
    except Exception as e:
        return {
            "message": f"Failed to run test: {str(e)}",
            "status": "error"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Port 8001 for Student Module (Teacher uses 8000) 