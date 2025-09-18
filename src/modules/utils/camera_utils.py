"""
Camera utilities for webcam management and frame processing.
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Camera manager for handling webcam input with threading support.
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: float = 30.0,
        buffer_size: int = 10
    ):
        """
        Initialize camera manager.
        
        Args:
            camera_id: Camera device ID
            resolution: Camera resolution (width, height)
            fps: Target frames per second
            buffer_size: Frame buffer size
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.buffer_size = buffer_size
        
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        
        # Statistics
        self.frames_captured = 0
        self.frames_dropped = 0
        self.start_time = None
    
    def start(self) -> bool:
        """Start camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_running = True
            self.start_time = time.time()
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logger.info(f"Camera {self.camera_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def stop(self):
        """Stop camera capture."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Camera stopped")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                self.frames_captured += 1
                
                # Add frame to queue
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Remove oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                        self.frames_dropped += 1
                    except queue.Empty:
                        pass
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                break
    
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get the latest frame."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get camera statistics."""
        if self.start_time is None:
            return {}
        
        elapsed_time = time.time() - self.start_time
        actual_fps = self.frames_captured / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'elapsed_time': elapsed_time,
            'actual_fps': actual_fps,
            'target_fps': self.fps,
            'queue_size': self.frame_queue.qsize()
        }


class FrameProcessor:
    """
    Frame processor for common image processing operations.
    """
    
    @staticmethod
    def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frame to target size."""
        return cv2.resize(frame, target_size)
    
    @staticmethod
    def enhance_contrast(frame: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
        """Enhance frame contrast."""
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    @staticmethod
    def normalize_lighting(frame: np.ndarray) -> np.ndarray:
        """Normalize lighting using CLAHE."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def detect_motion(
        frame1: np.ndarray, 
        frame2: np.ndarray, 
        threshold: int = 30
    ) -> Tuple[bool, float]:
        """Detect motion between two frames."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate motion percentage
        motion_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_percentage = motion_pixels / total_pixels
        
        # Determine if significant motion detected
        has_motion = motion_percentage > 0.01  # 1% threshold
        
        return has_motion, motion_percentage
