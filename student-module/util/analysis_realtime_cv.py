"""
Program constructs Concentration Index and returns a classification of engagement using MediaPipe.
"""

import cv2
import numpy as np
import mediapipe as mp
from math import hypot
from keras.models import load_model
import os

class analysis:
    """Face analysis class with emotion recognition and concentration tracking."""

    def __init__(self, identity_verifier=None):
        # Store identity verifier
        self.identity_verifier = identity_verifier

        # Load emotion model with error handling
        try:
            self.emotion_model = load_model('./util/model/emotion_recognition.h5')
        except Exception as e:
            print(f"Warning: Could not load emotion model: {e}")
            self.emotion_model = None

        # Initialize MediaPipe Face Mesh
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            raise

        # Initialize variables
        self.x = 0
        self.y = 0
        self.emotion = 5  # Default to Neutral
        self.size = 0
        self.frame_count = 0
        self.person_name = "Unknown"
        self.ear_threshold = 0.2
        
        # MediaPipe Face Mesh indices for eyes
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    def get_ear(self, landmarks):
        """Calculate Eye Aspect Ratio (EAR) using MediaPipe landmarks."""
        # Get left eye landmarks
        left_eye = np.array([(landmarks.landmark[point].x, landmarks.landmark[point].y) 
                            for point in self.LEFT_EYE])
        
        # Get right eye landmarks
        right_eye = np.array([(landmarks.landmark[point].x, landmarks.landmark[point].y) 
                             for point in self.RIGHT_EYE])
        
        # Calculate EAR for both eyes
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        
        # Return average EAR
        return (left_ear + right_ear) / 2.0
        
    def _calculate_ear(self, eye):
        """Calculate EAR for a single eye."""
        # Compute vertical distances
        v1 = hypot(eye[1][0] - eye[7][0], eye[1][1] - eye[7][1])
        v2 = hypot(eye[2][0] - eye[6][0], eye[2][1] - eye[6][1])
        v3 = hypot(eye[3][0] - eye[5][0], eye[3][1] - eye[5][1])
        
        # Compute horizontal distance
        h = hypot(eye[0][0] - eye[4][0], eye[0][1] - eye[4][1])
        
        # Calculate EAR
        ear = (v1 + v2 + v3) / (3.0 * h)
        return ear

    def get_gaze_ratio(self, frame, landmarks):
        """Calculate gaze ratio using MediaPipe landmarks."""
        # Get left eye landmarks
        left_eye = np.array([(landmarks.landmark[point].x, landmarks.landmark[point].y) 
                            for point in self.LEFT_EYE])
        
        # Convert to pixel coordinates
        h, w = frame.shape[:2]
        left_eye = np.array([(int(x * w), int(y * h)) for x, y in left_eye])
        
        # Create mask for eye region
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [left_eye], 255)
        
        # Get eye region
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Get eye region coordinates
        min_x = np.min(left_eye[:, 0])
        max_x = np.max(left_eye[:, 0])
        min_y = np.min(left_eye[:, 1])
        max_y = np.max(left_eye[:, 1])
        
        # Extract and threshold eye region
        eye_region = eye[min_y:max_y, min_x:max_x]
        _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)
        
        # Calculate gaze ratio
        height, width = threshold_eye.shape
        left_side = threshold_eye[0:height, 0:int(width/2)]
        right_side = threshold_eye[0:height, int(width/2):width]
        
        left_white = cv2.countNonZero(left_side)
        right_white = cv2.countNonZero(right_side)
        
        if right_white == 0:
            return 1.0
        return left_white / right_white

    def detect_face(self, frame):
        """
        Detect face and calculate concentration index using MediaPipe
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Get EAR for blink detection
                ear = self.get_ear(face_landmarks)
                self.size = ear
                
                # Get gaze ratio
                gaze_ratio = self.get_gaze_ratio(frame, face_landmarks)
                self.x = gaze_ratio
                
                # Detect emotion
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.detect_emotion(gray)

                # Perform face recognition if identity verifier is available
                if self.identity_verifier is not None:
                    try:
                        success, name = self.identity_verifier.check(frame)
                        self.person_name = name
                        print(f"[ANALYSIS] Identity check: success={success}, name={name}")
                    except Exception as e:
                        print(f"[ANALYSIS] Identity verification error: {e}")
                        self.person_name = "ERROR"

                # Get concentration index
                ci = self.gen_concentration_index()
                
                # Get gaze direction
                if gaze_ratio < 0.4:
                    gaze_direction = "LEFT"
                elif gaze_ratio > 0.6:
                    gaze_direction = "RIGHT"
                else:
                    gaze_direction = "CENTER"
                
                # Get emotion text
                emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',
                           3: 'Sad', 4: 'Surprised', 5: 'Neutral'}
                emotion_text = emotions.get(self.emotion, "Unknown")
                
                # Create overlay for text
                overlay = frame.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Add text with smaller font size and better positioning
                cv2.putText(overlay, f"Name: {self.person_name}",
                           (10, 30), font, 0.7, (0, 0, 255), 2)
                cv2.putText(overlay, f"Emotion: {emotion_text}",
                           (10, 60), font, 0.7, (0, 0, 255), 2)
                cv2.putText(overlay, f"Status: {ci}",
                           (10, 90), font, 0.7, (0, 0, 255), 2)
                cv2.putText(overlay, f"Gaze: {gaze_direction}",
                           (10, 120), font, 0.7, (0, 0, 255), 2)
                cv2.putText(overlay, f"EAR: {self.size:.2f}",
                           (10, 150), font, 0.7, (0, 0, 255), 2)
                
                # Apply the overlay with transparency
                alpha = 0.7
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
        return frame

    def detect_emotion(self, gray):
        """Detect emotion using the emotion recognition model."""
        if self.emotion_model is None:
            return  # Skip emotion detection if model failed to load
            
        emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',
                   3: 'Sad', 4: 'Surprised', 5: 'Neutral'}
        
        try:
            # Use MediaPipe face detection for emotion
            results = self.face_mesh.process(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get face bounding box
                    h, w = gray.shape
                    x_min = w
                    x_max = 0
                    y_min = h
                    y_max = 0
                    
                    for landmark in face_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        x_min = min(x_min, x)
                        x_max = max(x_max, x)
                        y_min = min(y_min, y)
                        y_max = max(y_max, y)
                    
                    # Add padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    # Extract face region
                    face_roi = gray[y_min:y_max, x_min:x_max]
                    
                    if face_roi.size > 0:
                        # Resize for emotion model
                        face_roi = cv2.resize(face_roi, (48, 48))
                        face_roi = face_roi.reshape([-1, 48, 48, 1])
                        face_roi = np.multiply(face_roi, 1.0 / 255.0)
                        
                        # Process emotion less frequently for performance
                        if self.frame_count % 10 == 0:
                            probab = self.emotion_model.predict(face_roi)[0] * 100
                            label = np.argmax(probab)
                            self.frame_count = 0
                            self.emotion = label
                            
            self.frame_count += 1
        except Exception as e:
            print(f"Warning: Emotion detection failed: {e}")
            self.frame_count += 1

    def gen_concentration_index(self):
        """Generate concentration index based on emotion and gaze."""
        # Emotion weights
        emotionweights = {0: 0.25, 1: 0.3, 2: 0.6,
                         3: 0.3, 4: 0.6, 5: 0.9}
        
        # Calculate gaze weights
        gaze_weights = 0
        if self.size < 0.15:  # More lenient eye closure threshold
            gaze_weights = 0
        elif self.size > 0.15 and self.size < 0.25:  # More lenient semi-closed threshold
            gaze_weights = 1.5
        else:
            if self.x < 2.2 and self.x > 0.8:  # More lenient center gaze threshold
                gaze_weights = 5
            else:
                gaze_weights = 2

        # Calculate concentration index
        concentration_index = (emotionweights[self.emotion] * gaze_weights) / 4.5
        
        # Return concentration level with more lenient thresholds
        if concentration_index > 0.5:  # Lowered from 0.65
            return "Highly Engaged"
        elif concentration_index > 0.2:  # Lowered from 0.25
            return "Engaged"
        else:
            return "Distracted"
            
    def set_person_name(self, name):
        """Set the name of the detected person."""
        self.person_name = name 