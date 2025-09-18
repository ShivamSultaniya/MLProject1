"""
OpenCV-based Face Detection Utility Module
Alternative to MediaPipe for ARM64 compatibility
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
import urllib.request

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False


class OpenCVFaceDetector:
    """Face detection using OpenCV DNN and dlib for landmarks"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        
        # Initialize OpenCV DNN face detector
        self.net = None
        self._load_face_detection_model()
        
        # Initialize dlib landmark predictor
        self.predictor = None
        self._load_landmark_predictor()
        
        # Eye landmark indices for dlib 68-point model
        self.LEFT_EYE_INDICES = list(range(36, 42))   # Points 36-41
        self.RIGHT_EYE_INDICES = list(range(42, 48))  # Points 42-47
        
    def _load_face_detection_model(self):
        """Load OpenCV DNN face detection model"""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        prototxt_path = os.path.join(model_dir, "deploy.prototxt")
        model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        
        # Download model files if they don't exist
        if not os.path.exists(prototxt_path):
            print("Downloading face detection prototxt...")
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            try:
                urllib.request.urlretrieve(prototxt_url, prototxt_path)
            except Exception as e:
                print(f"Failed to download prototxt: {e}")
                # Create a basic prototxt content
                self._create_basic_prototxt(prototxt_path)
        
        if not os.path.exists(model_path):
            print("Downloading face detection model (this may take a while)...")
            model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            try:
                urllib.request.urlretrieve(model_url, model_path)
            except Exception as e:
                print(f"Failed to download model: {e}")
                print("Using Haar Cascade as fallback...")
                self.use_haar_cascade = True
                return
        
        try:
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            self.use_haar_cascade = False
            print("OpenCV DNN face detector loaded successfully")
        except Exception as e:
            print(f"Failed to load DNN model: {e}")
            print("Using Haar Cascade as fallback...")
            self.use_haar_cascade = True
            self._load_haar_cascade()
    
    def _create_basic_prototxt(self, path: str):
        """Create a basic prototxt file for face detection"""
        prototxt_content = """name: "OpenCVFaceDetector"
input: "data"
input_dim: 1
input_dim: 3
input_dim: 300
input_dim: 300
layer {
  name: "detection_out"
  type: "DetectionOutput"
  bottom: "mbox_loc"
  bottom: "mbox_conf_flatten"
  bottom: "mbox_priorbox"
  top: "detection_out"
  include {
    phase: TEST
  }
  detection_output_param {
    num_classes: 2
    share_location: true
    background_label_id: 0
    nms_param {
      nms_threshold: 0.45
      top_k: 400
    }
    code_type: CENTER_SIZE
    keep_top_k: 200
    confidence_threshold: 0.02
  }
}"""
        with open(path, 'w') as f:
            f.write(prototxt_content)
    
    def _load_haar_cascade(self):
        """Load Haar Cascade as fallback"""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("Haar Cascade face detector loaded as fallback")
        except Exception as e:
            print(f"Failed to load Haar Cascade: {e}")
            self.face_cascade = None
    
    def _load_landmark_predictor(self):
        """Load dlib facial landmark predictor"""
        if not DLIB_AVAILABLE:
            print("Dlib not available, using simplified eye detection...")
            self.predictor = None
            return
            
        model_dir = "models"
        predictor_path = os.path.join(model_dir, "shape_predictor_68_face_landmarks.dat")
        
        if not os.path.exists(predictor_path):
            print("Facial landmark predictor not found.")
            print("Please download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("Extract and place in the 'models' directory")
            print("Using simplified eye detection as fallback...")
            self.predictor = None
            return
        
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
            print("Dlib facial landmark predictor loaded successfully")
        except Exception as e:
            print(f"Failed to load landmark predictor: {e}")
            print("Using simplified eye detection as fallback...")
            self.predictor = None
    
    def detect_faces(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect faces in the frame
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (face_detected, face_rectangle)
        """
        if hasattr(self, 'use_haar_cascade') and self.use_haar_cascade:
            return self._detect_faces_haar(frame)
        else:
            return self._detect_faces_dnn(frame)
    
    def _detect_faces_dnn(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Detect faces using OpenCV DNN"""
        if self.net is None:
            return False, None
        
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                x1 = min(w, x1)
                y1 = min(h, y1)
                
                return True, np.array([x, y, x1 - x, y1 - y])
        
        return False, None
    
    def _detect_faces_haar(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Detect faces using Haar Cascade"""
        if self.face_cascade is None:
            return False, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Return the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            return True, largest_face
        
        return False, None
    
    def get_eye_landmarks(self, face_rect: np.ndarray, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract eye landmarks from face rectangle
        
        Args:
            face_rect: Face rectangle [x, y, w, h]
            frame: Input frame
            
        Returns:
            Tuple of (left_eye_landmarks, right_eye_landmarks)
        """
        if self.predictor is not None:
            return self._get_eye_landmarks_dlib(face_rect, frame)
        else:
            return self._get_eye_landmarks_simple(face_rect, frame)
    
    def _get_eye_landmarks_dlib(self, face_rect: np.ndarray, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get eye landmarks using dlib"""
        if not DLIB_AVAILABLE:
            return self._get_eye_landmarks_simple(face_rect, frame)
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = face_rect
        
        # Convert to dlib rectangle
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        
        # Get facial landmarks
        landmarks = self.predictor(gray, dlib_rect)
        
        # Extract eye points
        left_eye = []
        right_eye = []
        
        for i in self.LEFT_EYE_INDICES:
            point = landmarks.part(i)
            left_eye.append([point.x, point.y])
        
        for i in self.RIGHT_EYE_INDICES:
            point = landmarks.part(i)
            right_eye.append([point.x, point.y])
        
        return np.array(left_eye), np.array(right_eye)
    
    def _get_eye_landmarks_simple(self, face_rect: np.ndarray, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple eye landmark estimation without dlib"""
        x, y, w, h = face_rect
        
        # Estimate eye positions based on face geometry
        # Left eye (from viewer's perspective)
        left_eye_x = x + int(w * 0.25)
        left_eye_y = y + int(h * 0.35)
        left_eye_w = int(w * 0.15)
        left_eye_h = int(h * 0.1)
        
        # Right eye
        right_eye_x = x + int(w * 0.6)
        right_eye_y = y + int(h * 0.35)
        right_eye_w = int(w * 0.15)
        right_eye_h = int(h * 0.1)
        
        # Create simple rectangular eye landmarks
        left_eye = np.array([
            [left_eye_x, left_eye_y],
            [left_eye_x, left_eye_y + left_eye_h],
            [left_eye_x + left_eye_w, left_eye_y + left_eye_h],
            [left_eye_x + left_eye_w, left_eye_y],
            [left_eye_x + left_eye_w//2, left_eye_y + left_eye_h//2],
            [left_eye_x + left_eye_w//2, left_eye_y + left_eye_h//2]
        ])
        
        right_eye = np.array([
            [right_eye_x, right_eye_y],
            [right_eye_x, right_eye_y + right_eye_h],
            [right_eye_x + right_eye_w, right_eye_y + right_eye_h],
            [right_eye_x + right_eye_w, right_eye_y],
            [right_eye_x + right_eye_w//2, right_eye_y + right_eye_h//2],
            [right_eye_x + right_eye_w//2, right_eye_y + right_eye_h//2]
        ])
        
        return left_eye, right_eye
    
    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        
        Args:
            eye_landmarks: Array of eye landmark points
            
        Returns:
            Eye aspect ratio value
        """
        if len(eye_landmarks) < 6:
            return 0.3  # Default EAR value
        
        if self.predictor is not None:
            # Use proper EAR calculation for dlib landmarks
            # Calculate distances between vertical eye landmarks
            vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            
            # Calculate distance between horizontal eye landmarks
            horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            # Calculate EAR
            if horizontal > 0:
                ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
                return ear
        else:
            # Simple EAR estimation for rectangular landmarks
            width = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            height = np.linalg.norm(eye_landmarks[1] - eye_landmarks[0])
            
            if width > 0:
                return height / width
        
        return 0.3  # Default EAR value
    
    def draw_landmarks(self, frame: np.ndarray, face_rect: np.ndarray) -> np.ndarray:
        """
        Draw face rectangle and eye regions on the frame
        
        Args:
            frame: Input frame
            face_rect: Face rectangle [x, y, w, h]
            
        Returns:
            Frame with drawn landmarks
        """
        annotated_frame = frame.copy()
        
        if face_rect is not None:
            x, y, w, h = face_rect.astype(int)
            
            # Draw face rectangle
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw eye regions
            left_eye_x = x + int(w * 0.25)
            left_eye_y = y + int(h * 0.35)
            left_eye_w = int(w * 0.15)
            left_eye_h = int(h * 0.1)
            
            right_eye_x = x + int(w * 0.6)
            right_eye_y = y + int(h * 0.35)
            right_eye_w = int(w * 0.15)
            right_eye_h = int(h * 0.1)
            
            cv2.rectangle(annotated_frame, (left_eye_x, left_eye_y), 
                         (left_eye_x + left_eye_w, left_eye_y + left_eye_h), (255, 0, 0), 2)
            cv2.rectangle(annotated_frame, (right_eye_x, right_eye_y), 
                         (right_eye_x + right_eye_w, right_eye_y + right_eye_h), (255, 0, 0), 2)
        
        return annotated_frame

