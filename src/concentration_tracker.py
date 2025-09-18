"""
Main Concentration Tracker Module
Integrates webcam, face detection, and concentration analysis
"""

import cv2
import yaml
import time
import logging
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.face_detector import FaceDetector
    use_mediapipe = True
except ImportError:
    print("MediaPipe not available, using OpenCV face detector...")
    from utils.opencv_face_detector import OpenCVFaceDetector as FaceDetector
    use_mediapipe = False

from utils.concentration_analyzer import ConcentrationAnalyzer


class ConcentrationTracker:
    """Main concentration tracking application"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the concentration tracker
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.face_detector = FaceDetector(
            confidence_threshold=self.config['face_detection']['confidence_threshold']
        )
        
        self.concentration_analyzer = ConcentrationAnalyzer(
            time_window=self.config['concentration']['time_window'],
            blink_threshold=self.config['eye_tracking']['blink_threshold'],
            drowsiness_threshold=self.config['eye_tracking']['drowsiness_threshold'],
            alert_threshold=self.config['concentration']['alert_threshold'],
            good_concentration_threshold=self.config['concentration']['good_concentration_threshold']
        )
        
        # Initialize camera
        self.cap = None
        self._setup_camera()
        
        # Tracking variables
        self.running = False
        self.start_time = None
        self.frame_count = 0
        self.last_log_time = time.time()
        
        self.logger.info("Concentration Tracker initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config', 'settings.yaml'
            )
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            print("Using default configuration...")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration...")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'camera': {'device_id': 0, 'width': 640, 'height': 480, 'fps': 30},
            'face_detection': {'confidence_threshold': 0.7},
            'eye_tracking': {'blink_threshold': 0.25, 'drowsiness_threshold': 0.3},
            'concentration': {'time_window': 30, 'alert_threshold': 0.4, 'good_concentration_threshold': 0.7},
            'logging': {'level': 'INFO', 'log_file': 'data/concentration_log.txt'},
            'display': {'show_video': True, 'show_landmarks': True, 'show_concentration_score': True, 'window_name': 'Concentration Tracker'}
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config['logging']['level'].upper())
        
        # Create data directory if it doesn't exist
        log_file = self.config['logging']['log_file']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(self.config['camera']['device_id'])
            
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
            self.cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            
            self.logger.info("Camera initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            raise
    
    def start_tracking(self):
        """Start the concentration tracking process"""
        if not self.cap or not self.cap.isOpened():
            self.logger.error("Camera not available")
            return
        
        self.running = True
        self.start_time = time.time()
        self.logger.info("Starting concentration tracking...")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    continue
                
                # Process frame
                results = self._process_frame(frame)
                
                # Display results
                if self.config['display']['show_video']:
                    display_frame = self._create_display_frame(frame, results)
                    cv2.imshow(self.config['display']['window_name'], display_frame)
                
                # Log results periodically
                self._log_results(results)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_tracking()
                elif key == ord('s'):
                    self._save_screenshot(frame, results)
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            self.logger.info("Tracking interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during tracking: {e}")
        finally:
            self.stop_tracking()
    
    def _process_frame(self, frame) -> Dict:
        """Process a single frame for concentration analysis"""
        # Detect face and get landmarks/face_rect
        face_detected, face_data = self.face_detector.detect_faces(frame)
        
        ear_left = 0.0
        ear_right = 0.0
        
        if face_detected and face_data is not None:
            if use_mediapipe:
                # MediaPipe returns landmarks
                left_eye, right_eye = self.face_detector.get_eye_landmarks(face_data, frame.shape)
            else:
                # OpenCV returns face rectangle
                left_eye, right_eye = self.face_detector.get_eye_landmarks(face_data, frame)
            
            if len(left_eye) > 0 and len(right_eye) > 0:
                ear_left = self.face_detector.calculate_eye_aspect_ratio(left_eye)
                ear_right = self.face_detector.calculate_eye_aspect_ratio(right_eye)
        
        # Update concentration analysis
        analysis_results = self.concentration_analyzer.update(ear_left, ear_right, face_detected)
        
        # Combine results
        results = {
            'face_detected': face_detected,
            'face_data': face_data,  # Could be landmarks or face_rect
            'ear_left': ear_left,
            'ear_right': ear_right,
            'timestamp': time.time(),
            **analysis_results
        }
        
        return results
    
    def _create_display_frame(self, frame, results: Dict):
        """Create display frame with annotations"""
        display_frame = frame.copy()
        
        # Draw face landmarks if enabled
        if (self.config['display']['show_landmarks'] and 
            results['face_detected'] and results['face_data'] is not None):
            if use_mediapipe:
                display_frame = self.face_detector.draw_landmarks(display_frame, results['face_data'])
            else:
                display_frame = self.face_detector.draw_landmarks(display_frame, results['face_data'])
        
        # Add text information
        if self.config['display']['show_concentration_score']:
            self._add_text_overlay(display_frame, results)
        
        return display_frame
    
    def _add_text_overlay(self, frame, results: Dict):
        """Add text overlay with concentration information"""
        h, w = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text information
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        # Concentration score
        score = results['concentration_score']
        score_color = self._get_score_color(score)
        cv2.putText(frame, f"Concentration: {score:.2f}", (20, y_offset), 
                   font, font_scale, score_color, thickness)
        y_offset += 25
        
        # Concentration level
        cv2.putText(frame, f"Level: {results['concentration_level']}", (20, y_offset), 
                   font, font_scale, color, thickness)
        y_offset += 25
        
        # Attention status
        cv2.putText(frame, f"Status: {results['attention_status']}", (20, y_offset), 
                   font, font_scale, color, thickness)
        y_offset += 25
        
        # Blink rate
        cv2.putText(frame, f"Blink Rate: {results['blink_rate']:.1f}/min", (20, y_offset), 
                   font, font_scale, color, thickness)
        y_offset += 25
        
        # Session time
        if self.start_time:
            session_time = time.time() - self.start_time
            minutes = int(session_time // 60)
            seconds = int(session_time % 60)
            cv2.putText(frame, f"Session: {minutes:02d}:{seconds:02d}", (20, y_offset), 
                       font, font_scale, color, thickness)
    
    def _get_score_color(self, score: float):
        """Get color based on concentration score"""
        if score >= self.config['concentration']['good_concentration_threshold']:
            return (0, 255, 0)  # Green
        elif score >= 0.5:
            return (0, 255, 255)  # Yellow
        elif score >= self.config['concentration']['alert_threshold']:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red
    
    def _log_results(self, results: Dict):
        """Log results periodically"""
        current_time = time.time()
        
        # Log every 10 seconds
        if current_time - self.last_log_time >= 10:
            self.logger.info(
                f"Concentration: {results['concentration_score']:.2f} "
                f"({results['concentration_level']}) - "
                f"Blink Rate: {results['blink_rate']:.1f}/min - "
                f"Face: {'Yes' if results['face_detected'] else 'No'}"
            )
            
            # Log summary stats
            summary = self.concentration_analyzer.get_summary_stats()
            if summary:
                self.logger.info(
                    f"Session Stats - Avg: {summary['average_score']:.2f}, "
                    f"Trend: {summary['score_trend']}"
                )
            
            self.last_log_time = current_time
    
    def _reset_tracking(self):
        """Reset tracking data"""
        self.concentration_analyzer = ConcentrationAnalyzer(
            time_window=self.config['concentration']['time_window'],
            blink_threshold=self.config['eye_tracking']['blink_threshold'],
            drowsiness_threshold=self.config['eye_tracking']['drowsiness_threshold'],
            alert_threshold=self.config['concentration']['alert_threshold'],
            good_concentration_threshold=self.config['concentration']['good_concentration_threshold']
        )
        self.start_time = time.time()
        self.frame_count = 0
        self.logger.info("Tracking data reset")
    
    def _save_screenshot(self, frame, results: Dict):
        """Save screenshot with current results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/screenshot_{timestamp}.jpg"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        display_frame = self._create_display_frame(frame, results)
        cv2.imwrite(filename, display_frame)
        
        self.logger.info(f"Screenshot saved: {filename}")
    
    def stop_tracking(self):
        """Stop the tracking process"""
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Log final summary
        if self.start_time:
            total_time = time.time() - self.start_time
            summary = self.concentration_analyzer.get_summary_stats()
            
            self.logger.info(f"Session completed - Duration: {total_time:.1f}s, Frames: {self.frame_count}")
            if summary:
                self.logger.info(
                    f"Final Stats - Current: {summary['current_score']:.2f}, "
                    f"Average: {summary['average_score']:.2f}, "
                    f"Trend: {summary['score_trend']}"
                )
        
        self.logger.info("Concentration tracking stopped")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Concentration Tracker Using Webcam")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file",
        default=None
    )
    parser.add_argument(
        "--no-display", 
        action="store_true", 
        help="Run without video display"
    )
    
    args = parser.parse_args()
    
    try:
        # Create tracker
        tracker = ConcentrationTracker(config_path=args.config)
        
        # Override display setting if specified
        if args.no_display:
            tracker.config['display']['show_video'] = False
        
        print("Concentration Tracker Starting...")
        print("Press 'q' to quit, 'r' to reset, 's' to save screenshot")
        
        # Start tracking
        tracker.start_tracking()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
