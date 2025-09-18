"""
Face Detection Utility Module
Handles face detection using MediaPipe and OpenCV
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """Face detection using MediaPipe Face Mesh"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices for MediaPipe Face Mesh
        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
    def detect_faces(self, frame: np.ndarray) -> Tuple[bool, Optional[List]]:
        """
        Detect faces in the frame
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (face_detected, landmarks)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            return True, results.multi_face_landmarks[0]
        return False, None
    
    def get_eye_landmarks(self, landmarks, frame_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract eye landmarks from face landmarks
        
        Args:
            landmarks: Face landmarks from MediaPipe
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Tuple of (left_eye_landmarks, right_eye_landmarks)
        """
        h, w = frame_shape[:2]
        
        left_eye = []
        right_eye = []
        
        for idx in self.LEFT_EYE_INDICES:
            landmark = landmarks.landmark[idx]
            left_eye.append([int(landmark.x * w), int(landmark.y * h)])
            
        for idx in self.RIGHT_EYE_INDICES:
            landmark = landmarks.landmark[idx]
            right_eye.append([int(landmark.x * w), int(landmark.y * h)])
            
        return np.array(left_eye), np.array(right_eye)
    
    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        
        Args:
            eye_landmarks: Array of eye landmark points
            
        Returns:
            Eye aspect ratio value
        """
        if len(eye_landmarks) < 6:
            return 0.0
            
        # Calculate distances between vertical eye landmarks
        vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Calculate distance between horizontal eye landmarks
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Calculate EAR
        if horizontal > 0:
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        return 0.0
    
    def draw_landmarks(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """
        Draw face landmarks on the frame
        
        Args:
            frame: Input frame
            landmarks: Face landmarks
            
        Returns:
            Frame with drawn landmarks
        """
        annotated_frame = frame.copy()
        
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            landmarks,
            self.mp_face_mesh.FACEMESH_CONTOURS,
            None,
            self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        
        return annotated_frame

