"""
Gaze tracking module using OpenCV.
"""

import cv2
import numpy as np

class GazeMonitor:
    """
    Monitors user's gaze direction using webcam feed.
    """
    
    def __init__(self):
        """Initialize the gaze tracker."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def get_eye_ratio(self, eye_img):
        """Calculate the ratio of white pixels in left vs right side of eye."""
        height, width = eye_img.shape
        left_side = eye_img[0:height, 0:int(width/2)]
        right_side = eye_img[0:height, int(width/2):width]
        
        left_white = cv2.countNonZero(left_side)
        right_white = cv2.countNonZero(right_side)
        
        if right_white == 0:
            return 1
        return left_white / right_white
        
    def dir(self, frame):
        """
        Determine gaze direction from frame.
        
        Args:
            frame: BGR image frame from OpenCV
            
        Returns:
            str: One of "LOOKING_AWAY", "LOOKING_CENTER", "LOOKING_LEFT", "LOOKING_RIGHT"
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return "LOOKING_AWAY"
            
        for (x, y, w, h) in faces:
            # Region of interest for eyes
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 0:
                return "LOOKING_AWAY"
                
            for (ex, ey, ew, eh) in eyes:
                # Get eye region
                eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_roi = cv2.resize(eye_roi, (100, 50))
                
                # Threshold eye image
                _, eye_roi = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY)
                
                # Calculate gaze ratio
                gaze_ratio = self.get_eye_ratio(eye_roi)
                
                # Define thresholds for left/right/center
                if gaze_ratio < 0.4:
                    return "LOOKING_LEFT"
                elif gaze_ratio > 0.6:
                    return "LOOKING_RIGHT"
                else:
                    return "LOOKING_CENTER"
                    
        return "LOOKING_AWAY" 