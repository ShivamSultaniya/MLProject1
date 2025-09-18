"""
Webcam Interface for Real-time Concentration Analysis

Handles webcam capture, processing, and display for the concentration analysis system.
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Callable, Dict, Any
import logging
from queue import Queue, Empty


class WebcamInterface:
    """
    Real-time webcam interface for concentration analysis.
    """
    
    def __init__(self, camera_id: int = 0, resolution: tuple = (640, 480),
                 fps: int = 30, buffer_size: int = 10):
        """
        Initialize webcam interface.
        
        Args:
            camera_id: Camera device ID
            resolution: Camera resolution (width, height)
            fps: Target frames per second
            buffer_size: Frame buffer size
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.target_fps = fps
        self.buffer_size = buffer_size
        
        # Video capture
        self.cap = None
        self.is_running = False
        
        # Threading
        self.capture_thread = None
        self.frame_queue = Queue(maxsize=buffer_size)
        
        # Performance tracking
        self.actual_fps = 0
        self.frame_count = 0
        self.start_time = None
        
        # Callbacks
        self.frame_callback = None
        self.analysis_callback = None
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_camera(self) -> bool:
        """
        Initialize camera capture.
        
        Returns:
            True if successful
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                self.logger.error(f"Cannot open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False
    
    def start_capture(self) -> bool:
        """
        Start video capture in separate thread.
        
        Returns:
            True if started successfully
        """
        if not self.initialize_camera():
            return False
        
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.logger.info("Video capture started")
        return True
    
    def stop_capture(self):
        """Stop video capture."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        self.logger.info("Video capture stopped")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    time.sleep(0.01)
                    continue
                
                # Add timestamp to frame
                timestamp = time.time()
                frame_data = {
                    'frame': frame,
                    'timestamp': timestamp,
                    'frame_id': self.frame_count
                }
                
                # Add to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame_data)
                except:
                    # Queue full, drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                    except Empty:
                        pass
                
                self.frame_count += 1
                
                # Calculate FPS
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # Frame rate limiting
                time.sleep(1.0 / self.target_fps)
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                break
    
    def get_frame(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        Get the latest frame from the queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Frame data dictionary or None
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def set_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback function for raw frames."""
        self.frame_callback = callback
    
    def set_analysis_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback function for analysis results."""
        self.analysis_callback = callback
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information."""
        if not self.cap:
            return {}
        
        return {
            'camera_id': self.camera_id,
            'resolution': (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'actual_fps': self.actual_fps,
            'frame_count': self.frame_count,
            'is_running': self.is_running
        }
    
    def save_frame(self, frame: np.ndarray, filepath: str) -> bool:
        """
        Save frame to file.
        
        Args:
            frame: Frame to save
            filepath: Output file path
            
        Returns:
            True if successful
        """
        try:
            cv2.imwrite(filepath, frame)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save frame: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()

