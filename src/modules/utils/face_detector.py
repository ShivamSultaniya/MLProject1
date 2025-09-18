"""
Face detection and landmark extraction utilities.
"""

import cv2
import numpy as np
import dlib
import mediapipe as mp
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Unified face detection and landmark extraction using multiple backends.
    Supports OpenCV Haar cascades, dlib, and MediaPipe.
    """
    
    def __init__(
        self,
        backend: str = 'mediapipe',  # 'opencv', 'dlib', 'mediapipe'
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize face detector.
        
        Args:
            backend: Detection backend to use
            model_path: Path to model files (for dlib)
            confidence_threshold: Minimum confidence for detections
        """
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        
        # Initialize backend
        if backend == 'opencv':
            self._init_opencv()
        elif backend == 'dlib':
            self._init_dlib(model_path)
        elif backend == 'mediapipe':
            self._init_mediapipe()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        logger.info(f"FaceDetector initialized with {backend} backend")
    
    def _init_opencv(self):
        """Initialize OpenCV face detector."""
        try:
            # Load Haar cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise RuntimeError("Could not load Haar cascade")
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV detector: {e}")
            raise
    
    def _init_dlib(self, model_path: Optional[str] = None):
        """Initialize dlib face detector."""
        try:
            # Face detector
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Landmark predictor
            if model_path:
                self.landmark_predictor = dlib.shape_predictor(model_path)
            else:
                # Try default path
                try:
                    self.landmark_predictor = dlib.shape_predictor(
                        'shape_predictor_68_face_landmarks.dat'
                    )
                except:
                    logger.warning("No landmark predictor available for dlib")
                    self.landmark_predictor = None
                    
        except Exception as e:
            logger.error(f"Failed to initialize dlib detector: {e}")
            raise
    
    def _init_mediapipe(self):
        """Initialize MediaPipe face detector."""
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Face detection model
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for close-range, 1 for full-range
                min_detection_confidence=self.confidence_threshold
            )
            
            # Face mesh model for landmarks
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.confidence_threshold,
                min_tracking_confidence=0.5
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe detector: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face bounding boxes [(x1, y1, x2, y2), ...]
        """
        if self.backend == 'opencv':
            return self._detect_faces_opencv(image)
        elif self.backend == 'dlib':
            return self._detect_faces_dlib(image)
        elif self.backend == 'mediapipe':
            return self._detect_faces_mediapipe(image)
        else:
            return []
    
    def _detect_faces_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to (x1, y1, x2, y2) format
        face_boxes = []
        for (x, y, w, h) in faces:
            face_boxes.append((x, y, x + w, y + h))
        
        return face_boxes
    
    def _detect_faces_dlib(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using dlib."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_detector(gray)
        
        face_boxes = []
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            face_boxes.append((x1, y1, x2, y2))
        
        return face_boxes
    
    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        face_boxes = []
        if results.detections:
            h, w = image.shape[:2]
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(w-1, x1))
                y1 = max(0, min(h-1, y1))
                x2 = max(0, min(w-1, x2))
                y2 = max(0, min(h-1, y2))
                
                if x2 > x1 and y2 > y1:  # Valid box
                    face_boxes.append((x1, y1, x2, y2))
        
        return face_boxes
    
    def get_landmarks(
        self, 
        image: np.ndarray, 
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[np.ndarray]:
        """
        Get facial landmarks.
        
        Args:
            image: Input image
            face_bbox: Face bounding box (if None, will detect automatically)
            
        Returns:
            Array of landmark points [(x, y), ...] or None if failed
        """
        if self.backend == 'dlib':
            return self._get_landmarks_dlib(image, face_bbox)
        elif self.backend == 'mediapipe':
            return self._get_landmarks_mediapipe(image)
        else:
            logger.warning(f"Landmark extraction not supported for {self.backend}")
            return None
    
    def _get_landmarks_dlib(
        self, 
        image: np.ndarray, 
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[np.ndarray]:
        """Get landmarks using dlib."""
        if self.landmark_predictor is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if face_bbox is None:
            # Detect face first
            faces = self._detect_faces_dlib(image)
            if not faces:
                return None
            face_bbox = faces[0]
        
        # Convert to dlib rectangle
        x1, y1, x2, y2 = face_bbox
        rect = dlib.rectangle(x1, y1, x2, y2)
        
        # Get landmarks
        landmarks = self.landmark_predictor(gray, rect)
        
        # Convert to numpy array
        points = []
        for i in range(landmarks.num_parts):
            point = landmarks.part(i)
            points.append([point.x, point.y])
        
        return np.array(points, dtype=np.int32)
    
    def _get_landmarks_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get landmarks using MediaPipe."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            h, w = image.shape[:2]
            
            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert to pixel coordinates
            points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append([x, y])
            
            return np.array(points, dtype=np.int32)
        
        return None
    
    def get_eye_landmarks(self, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract eye landmarks from full face landmarks.
        
        Args:
            landmarks: Full face landmarks array
            
        Returns:
            Tuple of (left_eye_landmarks, right_eye_landmarks)
        """
        if self.backend == 'dlib':
            # dlib 68-point model indices
            left_eye = landmarks[36:42]   # Left eye (6 points)
            right_eye = landmarks[42:48]  # Right eye (6 points)
        elif self.backend == 'mediapipe':
            # MediaPipe face mesh indices (approximate eye regions)
            # These are simplified - MediaPipe has many more eye landmarks
            left_eye_indices = [33, 7, 163, 144, 145, 153]  # Left eye approximation
            right_eye_indices = [362, 382, 381, 380, 374, 373]  # Right eye approximation
            
            left_eye = landmarks[left_eye_indices]
            right_eye = landmarks[right_eye_indices]
        else:
            raise ValueError(f"Eye landmark extraction not supported for {self.backend}")
        
        return left_eye, right_eye
    
    def visualize_detections(
        self, 
        image: np.ndarray, 
        faces: List[Tuple[int, int, int, int]], 
        landmarks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Visualize face detections and landmarks on image.
        
        Args:
            image: Input image
            faces: List of face bounding boxes
            landmarks: Optional landmarks to draw
            
        Returns:
            Image with visualizations
        """
        vis_image = image.copy()
        
        # Draw face bounding boxes
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw landmarks
        if landmarks is not None:
            for point in landmarks:
                cv2.circle(vis_image, tuple(point), 2, (0, 0, 255), -1)
            
            # Draw eye regions if we have enough landmarks
            if len(landmarks) >= 48:  # dlib 68-point model
                left_eye, right_eye = self.get_eye_landmarks(landmarks)
                
                # Draw eye contours
                cv2.polylines(vis_image, [left_eye], True, (255, 0, 0), 1)
                cv2.polylines(vis_image, [right_eye], True, (255, 0, 0), 1)
        
        return vis_image
    
    def get_face_region(
        self, 
        image: np.ndarray, 
        face_bbox: Tuple[int, int, int, int],
        expand_ratio: float = 0.1
    ) -> np.ndarray:
        """
        Extract face region from image with optional expansion.
        
        Args:
            image: Input image
            face_bbox: Face bounding box
            expand_ratio: Ratio to expand the bounding box
            
        Returns:
            Cropped face region
        """
        x1, y1, x2, y2 = face_bbox
        h, w = image.shape[:2]
        
        # Calculate expansion
        face_w = x2 - x1
        face_h = y2 - y1
        
        expand_w = int(face_w * expand_ratio)
        expand_h = int(face_h * expand_ratio)
        
        # Expand bounding box
        x1_exp = max(0, x1 - expand_w)
        y1_exp = max(0, y1 - expand_h)
        x2_exp = min(w, x2 + expand_w)
        y2_exp = min(h, y2 + expand_h)
        
        # Extract region
        face_region = image[y1_exp:y2_exp, x1_exp:x2_exp]
        
        return face_region
    
    def is_face_frontal(
        self, 
        landmarks: np.ndarray, 
        angle_threshold: float = 30.0
    ) -> bool:
        """
        Check if face is approximately frontal based on landmarks.
        
        Args:
            landmarks: Facial landmarks
            angle_threshold: Maximum angle deviation for frontal face
            
        Returns:
            True if face is approximately frontal
        """
        if len(landmarks) < 17:  # Need at least face contour points
            return False
        
        try:
            if self.backend == 'dlib':
                # Use nose tip and face contour points
                nose_tip = landmarks[30]
                left_face = landmarks[0]
                right_face = landmarks[16]
                
                # Calculate face center
                face_center_x = (left_face[0] + right_face[0]) / 2
                
                # Check if nose is approximately centered
                nose_offset = abs(nose_tip[0] - face_center_x)
                face_width = right_face[0] - left_face[0]
                
                offset_ratio = nose_offset / face_width if face_width > 0 else 1.0
                
                # Consider face frontal if nose offset is less than 15% of face width
                return offset_ratio < 0.15
            
            else:
                # Simplified check for other backends
                return True
                
        except Exception as e:
            logger.error(f"Error checking face frontality: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
