"""
Face identity verification module using OpenCV's face recognition.
"""

from pathlib import Path, PureWindowsPath
import numpy as np
import cv2

from .config import FACE_DATASET, ID_TOLERANCE

class IDVerifier:
    """
    Handles face identity verification using OpenCV's face recognition.
    """
    
    def __init__(self, split="train"):
        """
        Initialize the verifier by loading face encodings from the dataset.
        
        Args:
            split (str): Dataset split to use ('train' or 'test')
        """
        try:
            # Load face detection model
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Load face recognition model
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            base = Path(PureWindowsPath(FACE_DATASET)) / split
            if not base.exists():
                raise FileNotFoundError(f"Dataset directory not found: {base}")
                
            self.labels = []
            faces = []
            label_id = 0
            label_map = {}
        except Exception as e:
            print(f"Error initializing IDVerifier: {e}")
            raise
        
        # Load and process training images
        for student in base.iterdir():
            label_map[label_id] = student.name
            for img in student.glob("*.jpg"):
                try:
                    # Read and convert to grayscale
                    img_array = cv2.imread(str(img))
                    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    
                    # Detect face
                    face_rects = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, 
                        minNeighbors=5,
                        minSize=(30, 30)
                    )
                    
                    for (x, y, w, h) in face_rects:
                        face = gray[y:y+h, x:x+w]
                        faces.append(face)
                        self.labels.append(label_id)
                        
                except Exception as e:
                    print(f"Warning: Failed to process {img}: {e}")
            
            label_id += 1
            
        if faces:
            # Train the recognizer
            self.face_recognizer.train(faces, np.array(self.labels))
            self.label_map = label_map
            print(f"[ID] {len(faces)} faces loaded from {base}")
        else:
            print("[ID] No faces found in dataset")
            self.face_recognizer = None

    def check(self, bgr):
        """
        Check if the face in the frame matches any known identity.
        
        Args:
            bgr: BGR image frame from OpenCV
            
        Returns:
            tuple: (bool, str) - (match success, identity label)
        """
        if self.face_recognizer is None:
            return True, "DISABLED"
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            
            # Detect face
            face_rects = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(face_rects) == 0:
                return False, "NO_FACE"
                
            # Get the largest face
            face_rect = max(face_rects, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = face_rect
            face = gray[y:y+h, x:x+w]
            
            # Predict
            label_id, confidence = self.face_recognizer.predict(face)

            # Lower confidence is better in LBPH (0 = perfect match, higher = poor match)
            # Typical good matches are 30-70, unknown faces are 80-120+
            # ID_TOLERANCE of 0.50 means threshold of 100 (anything below 100 is accepted)
            threshold = ID_TOLERANCE * 200  # 0.50 * 200 = 100

            # Debug logging
            print(f"[FACE_RECOG] Confidence: {confidence:.2f}, Threshold: {threshold}, Label ID: {label_id}, Name: {self.label_map.get(label_id, 'INVALID')}")

            if confidence < threshold:
                recognized_name = self.label_map[label_id]
                print(f"[FACE_RECOG] ✓ RECOGNIZED: {recognized_name}")
                return True, recognized_name
            else:
                print(f"[FACE_RECOG] ✗ UNKNOWN (confidence {confidence:.2f} >= threshold {threshold})")
                return False, "UNKNOWN"
                
        except Exception as e:
            print(f"Warning: Face verification failed: {e}")
            return False, "ERROR"

if __name__ == "__main__":
    # Quick test to verify encodings are loaded
    verifier = IDVerifier()
    print(f"Loaded {len(verifier.labels)} face encodings") 