"""
Main concentration analyzer that integrates all modules.
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

from ..modules.eye_gaze import GazeEstimator
from ..modules.blink_drowsiness import BlinkDetector, DrowsinessDetector
from ..modules.head_pose import HeadPoseEstimator
from ..modules.engagement import EngagementDetector
from .feedback_system import FeedbackSystem
from .data_logger import DataLogger

logger = logging.getLogger(__name__)


class ConcentrationAnalyzer:
    """
    Main concentration analyzer that integrates all analysis modules
    to provide comprehensive real-time concentration monitoring.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_modules: Optional[List[str]] = None,
        device: str = 'auto',
        max_fps: float = 30.0,
        enable_feedback: bool = True,
        enable_logging: bool = True
    ):
        """
        Initialize concentration analyzer.
        
        Args:
            config: Configuration dictionary for all modules
            enable_modules: List of modules to enable
            device: Device for inference
            max_fps: Maximum processing FPS
            enable_feedback: Enable feedback system
            enable_logging: Enable data logging
        """
        self.config = config or {}
        self.device = device
        self.max_fps = max_fps
        self.enable_feedback = enable_feedback
        self.enable_logging = enable_logging
        
        # Default enabled modules
        if enable_modules is None:
            enable_modules = ['gaze', 'blink', 'drowsiness', 'pose', 'engagement']
        self.enable_modules = enable_modules
        
        # Initialize modules
        self.modules = {}
        self._initialize_modules()
        
        # Initialize feedback system
        if self.enable_feedback:
            self.feedback_system = FeedbackSystem(
                config=self.config.get('feedback', {})
            )
        
        # Initialize data logger
        if self.enable_logging:
            self.data_logger = DataLogger(
                config=self.config.get('logging', {})
            )
        
        # Processing state
        self.is_running = False
        self.processing_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=100)
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_times = deque(maxlen=100)
        
        # Results history
        self.concentration_history = deque(maxlen=300)  # 10 seconds at 30fps
        self.alert_history = deque(maxlen=1000)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"ConcentrationAnalyzer initialized with modules: {enable_modules}")
    
    def _initialize_modules(self):
        """Initialize all enabled analysis modules."""
        try:
            # Eye gaze estimation
            if 'gaze' in self.enable_modules:
                gaze_config = self.config.get('gaze', {})
                self.modules['gaze'] = GazeEstimator(
                    model_type=gaze_config.get('model_type', 'resnet'),
                    model_path=gaze_config.get('model_path'),
                    device=self.device
                )
                logger.info("Gaze estimation module initialized")
            
            # Blink detection
            if 'blink' in self.enable_modules:
                blink_config = self.config.get('blink', {})
                self.modules['blink'] = BlinkDetector(
                    method=blink_config.get('method', 'hybrid'),
                    model_path=blink_config.get('model_path'),
                    device=self.device
                )
                logger.info("Blink detection module initialized")
            
            # Drowsiness detection
            if 'drowsiness' in self.enable_modules:
                drowsiness_config = self.config.get('drowsiness', {})
                blink_detector = self.modules.get('blink')  # Reuse blink detector
                self.modules['drowsiness'] = DrowsinessDetector(
                    blink_detector=blink_detector,
                    model_path=drowsiness_config.get('model_path'),
                    device=self.device
                )
                logger.info("Drowsiness detection module initialized")
            
            # Head pose estimation
            if 'pose' in self.enable_modules:
                pose_config = self.config.get('pose', {})
                self.modules['pose'] = HeadPoseEstimator(
                    model_type=pose_config.get('model_type', 'resnet'),
                    model_path=pose_config.get('model_path'),
                    device=self.device
                )
                logger.info("Head pose estimation module initialized")
            
            # Engagement recognition
            if 'engagement' in self.enable_modules:
                engagement_config = self.config.get('engagement', {})
                self.modules['engagement'] = EngagementDetector(
                    model_type=engagement_config.get('model_type', 'cnn'),
                    model_path=engagement_config.get('model_path'),
                    device=self.device
                )
                logger.info("Engagement recognition module initialized")
                
        except Exception as e:
            logger.error(f"Error initializing modules: {e}")
            raise
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a single frame for concentration indicators.
        
        Args:
            frame: Input video frame
            
        Returns:
            Comprehensive analysis results
        """
        start_time = time.time()
        
        results = {
            'timestamp': start_time,
            'frame_number': self.frame_count,
            'gaze': None,
            'blink': None,
            'drowsiness': None,
            'pose': None,
            'engagement': None,
            'concentration_score': 0.0,
            'attention_level': 'unknown',
            'alerts': [],
            'processing_time': 0.0
        }
        
        try:
            # Create futures for parallel processing
            futures = {}
            
            # Gaze estimation
            if 'gaze' in self.modules:
                futures['gaze'] = self.executor.submit(
                    self.modules['gaze'].estimate_gaze, frame
                )
            
            # Head pose estimation
            if 'pose' in self.modules:
                futures['pose'] = self.executor.submit(
                    self.modules['pose'].estimate_pose, frame
                )
            
            # Blink detection (sequential as drowsiness depends on it)
            if 'blink' in self.modules:
                blink_results = self.modules['blink'].detect_blink(frame)
                results['blink'] = blink_results
            
            # Drowsiness detection
            if 'drowsiness' in self.modules:
                pose_info = None
                if 'pose' in futures:
                    # Wait for pose if available
                    try:
                        pose_info = futures['pose'].result(timeout=0.1)
                    except:
                        pass
                
                head_pose = None
                if pose_info and pose_info.get('valid', False):
                    head_pose = (pose_info['pitch'], pose_info['yaw'])
                
                drowsiness_results = self.modules['drowsiness'].analyze_drowsiness(
                    frame, head_pose=head_pose
                )
                results['drowsiness'] = drowsiness_results
            
            # Collect parallel results
            for module_name, future in futures.items():
                try:
                    results[module_name] = future.result(timeout=1.0)
                except Exception as e:
                    logger.warning(f"Module {module_name} processing failed: {e}")
                    results[module_name] = None
            
            # Engagement recognition (uses results from other modules)
            if 'engagement' in self.modules:
                engagement_results = self.modules['engagement'].detect_engagement(
                    frame,
                    gaze_info=results.get('gaze'),
                    pose_info=results.get('pose'),
                    blink_info=results.get('blink')
                )
                results['engagement'] = engagement_results
            
            # Calculate overall concentration score
            concentration_score = self._calculate_concentration_score(results)
            results['concentration_score'] = concentration_score
            
            # Determine attention level
            attention_level = self._classify_attention_level(concentration_score, results)
            results['attention_level'] = attention_level
            
            # Generate alerts
            alerts = self._generate_alerts(results)
            results['alerts'] = alerts
            
            # Update statistics
            self.frame_count += 1
            processing_time = time.time() - start_time
            results['processing_time'] = processing_time
            self.processing_times.append(processing_time)
            
            # Update history
            self.concentration_history.append(concentration_score)
            
            # Log alerts
            for alert in alerts:
                self.alert_history.append({
                    'timestamp': start_time,
                    'type': alert['type'],
                    'severity': alert['severity'],
                    'message': alert['message']
                })
            
            # Send to feedback system
            if self.enable_feedback and hasattr(self, 'feedback_system'):
                self.feedback_system.process_results(results)
            
            # Log data
            if self.enable_logging and hasattr(self, 'data_logger'):
                self.data_logger.log_results(results)
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            results['processing_time'] = time.time() - start_time
        
        return results
    
    def _calculate_concentration_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall concentration score from all modules."""
        score = 0.0
        weight_sum = 0.0
        
        try:
            # Gaze contribution (25%)
            gaze_results = results.get('gaze')
            if gaze_results and gaze_results.get('valid', False):
                gaze_confidence = gaze_results.get('confidence', 0.0)
                # Bonus for looking at center/screen
                gaze_angles = gaze_results.get('gaze_angles', (0, 0))
                if gaze_angles:
                    pitch, yaw = gaze_angles
                    center_bonus = max(0, 1 - (abs(pitch) + abs(yaw)) / 60.0)
                    gaze_score = gaze_confidence * (0.5 + 0.5 * center_bonus)
                else:
                    gaze_score = gaze_confidence
                
                score += gaze_score * 0.25
                weight_sum += 0.25
            
            # Engagement contribution (30%)
            engagement_results = results.get('engagement')
            if engagement_results:
                engagement_level = engagement_results.get('engagement_level')
                if engagement_level and hasattr(engagement_level, 'value'):
                    # Convert engagement level to score (0-1)
                    engagement_score = engagement_level.value / 3.0
                    engagement_confidence = engagement_results.get('engagement_confidence', 0.0)
                    
                    # Weight by confidence
                    weighted_engagement = engagement_score * engagement_confidence
                    score += weighted_engagement * 0.30
                    weight_sum += 0.30
            
            # Head pose contribution (20%)
            pose_results = results.get('pose')
            if pose_results and pose_results.get('valid', False):
                pose_confidence = pose_results.get('confidence', 0.0)
                pitch, yaw, roll = pose_results.get('pitch', 0), pose_results.get('yaw', 0), pose_results.get('roll', 0)
                
                # Penalty for looking away
                pose_penalty = 1.0 - (abs(yaw) + abs(pitch)) / 90.0
                pose_score = pose_confidence * max(0.0, pose_penalty)
                
                score += pose_score * 0.20
                weight_sum += 0.20
            
            # Drowsiness contribution (25%)
            drowsiness_results = results.get('drowsiness')
            if drowsiness_results:
                drowsiness_level = drowsiness_results.get('drowsiness_level')
                if drowsiness_level and hasattr(drowsiness_level, 'value'):
                    # Invert drowsiness (high drowsiness = low concentration)
                    alertness_score = 1.0 - (drowsiness_level.value / 4.0)
                    score += alertness_score * 0.25
                    weight_sum += 0.25
            
            # Normalize by actual weights
            if weight_sum > 0:
                score = score / weight_sum
            else:
                score = 0.5  # Default neutral score
            
        except Exception as e:
            logger.error(f"Error calculating concentration score: {e}")
            score = 0.5
        
        return max(0.0, min(1.0, score))
    
    def _classify_attention_level(
        self, 
        concentration_score: float, 
        results: Dict[str, Any]
    ) -> str:
        """Classify overall attention level."""
        try:
            # Primary classification based on concentration score
            if concentration_score >= 0.8:
                base_level = 'very_high'
            elif concentration_score >= 0.6:
                base_level = 'high'
            elif concentration_score >= 0.4:
                base_level = 'moderate'
            elif concentration_score >= 0.2:
                base_level = 'low'
            else:
                base_level = 'very_low'
            
            # Adjust based on specific indicators
            drowsiness_results = results.get('drowsiness')
            if drowsiness_results:
                drowsiness_level = drowsiness_results.get('drowsiness_level')
                if drowsiness_level and hasattr(drowsiness_level, 'value'):
                    if drowsiness_level.value >= 3:  # Very/extremely drowsy
                        return 'very_low'
                    elif drowsiness_level.value >= 2:  # Moderately drowsy
                        return min(base_level, 'low', key=lambda x: ['very_low', 'low', 'moderate', 'high', 'very_high'].index(x))
            
            # Check for distraction
            engagement_results = results.get('engagement')
            if engagement_results and engagement_results.get('distraction_detected', False):
                return min(base_level, 'moderate', key=lambda x: ['very_low', 'low', 'moderate', 'high', 'very_high'].index(x))
            
            return base_level
            
        except Exception as e:
            logger.error(f"Error classifying attention level: {e}")
            return 'moderate'
    
    def _generate_alerts(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on analysis results."""
        alerts = []
        
        try:
            concentration_score = results.get('concentration_score', 0.5)
            
            # Low concentration alert
            if concentration_score < 0.3:
                alerts.append({
                    'type': 'low_concentration',
                    'severity': 'high',
                    'message': f'Low concentration detected (score: {concentration_score:.2f})',
                    'timestamp': results['timestamp']
                })
            elif concentration_score < 0.5:
                alerts.append({
                    'type': 'moderate_concentration',
                    'severity': 'medium',
                    'message': f'Moderate concentration (score: {concentration_score:.2f})',
                    'timestamp': results['timestamp']
                })
            
            # Drowsiness alerts
            drowsiness_results = results.get('drowsiness')
            if drowsiness_results:
                drowsiness_level = drowsiness_results.get('drowsiness_level')
                if drowsiness_level and hasattr(drowsiness_level, 'value'):
                    if drowsiness_level.value >= 3:
                        alerts.append({
                            'type': 'high_drowsiness',
                            'severity': 'critical',
                            'message': f'High drowsiness detected: {drowsiness_level.name}',
                            'timestamp': results['timestamp']
                        })
                    elif drowsiness_level.value >= 2:
                        alerts.append({
                            'type': 'moderate_drowsiness',
                            'severity': 'high',
                            'message': f'Moderate drowsiness detected: {drowsiness_level.name}',
                            'timestamp': results['timestamp']
                        })
                
                # Microsleep alert
                if drowsiness_results.get('microsleep_detected', False):
                    alerts.append({
                        'type': 'microsleep',
                        'severity': 'critical',
                        'message': 'Microsleep episode detected',
                        'timestamp': results['timestamp']
                    })
            
            # Engagement alerts
            engagement_results = results.get('engagement')
            if engagement_results:
                if engagement_results.get('distraction_detected', False):
                    alerts.append({
                        'type': 'distraction',
                        'severity': 'medium',
                        'message': 'Distraction detected',
                        'timestamp': results['timestamp']
                    })
                
                engagement_level = engagement_results.get('engagement_level')
                if engagement_level and hasattr(engagement_level, 'value'):
                    if engagement_level.value == 0:  # Very low engagement
                        alerts.append({
                            'type': 'low_engagement',
                            'severity': 'high',
                            'message': 'Very low engagement detected',
                            'timestamp': results['timestamp']
                        })
            
            # Gaze alerts
            gaze_results = results.get('gaze')
            if gaze_results and gaze_results.get('valid', False):
                gaze_angles = gaze_results.get('gaze_angles', (0, 0))
                if gaze_angles:
                    pitch, yaw = gaze_angles
                    if abs(yaw) > 30 or abs(pitch) > 20:
                        alerts.append({
                            'type': 'looking_away',
                            'severity': 'medium',
                            'message': f'Looking away from screen (yaw: {yaw:.1f}°, pitch: {pitch:.1f}°)',
                            'timestamp': results['timestamp']
                        })
            
            # Head pose alerts
            pose_results = results.get('pose')
            if pose_results and pose_results.get('valid', False):
                yaw = pose_results.get('yaw', 0)
                if abs(yaw) > 45:
                    alerts.append({
                        'type': 'head_turned_away',
                        'severity': 'medium',
                        'message': f'Head turned away (yaw: {yaw:.1f}°)',
                        'timestamp': results['timestamp']
                    })
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
        
        return alerts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics."""
        current_time = time.time()
        session_duration = current_time - self.start_time
        
        # Processing statistics
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        current_fps = len(self.processing_times) / session_duration if session_duration > 0 else 0
        
        # Concentration statistics
        concentration_values = list(self.concentration_history)
        avg_concentration = np.mean(concentration_values) if concentration_values else 0.5
        
        # Alert statistics
        alert_counts = {}
        for alert in self.alert_history:
            alert_type = alert['type']
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        
        # Module statistics
        module_stats = {}
        for module_name, module in self.modules.items():
            if hasattr(module, 'get_statistics') or hasattr(module, 'get_pose_statistics') or hasattr(module, 'get_engagement_statistics'):
                try:
                    if hasattr(module, 'get_statistics'):
                        module_stats[module_name] = module.get_statistics()
                    elif hasattr(module, 'get_pose_statistics'):
                        module_stats[module_name] = module.get_pose_statistics()
                    elif hasattr(module, 'get_engagement_statistics'):
                        module_stats[module_name] = module.get_engagement_statistics()
                except Exception as e:
                    logger.warning(f"Could not get statistics from {module_name}: {e}")
        
        return {
            'session_duration': session_duration,
            'frames_processed': self.frame_count,
            'average_fps': current_fps,
            'average_processing_time': avg_processing_time,
            'average_concentration': avg_concentration,
            'current_concentration': concentration_values[-1] if concentration_values else 0.5,
            'alert_counts': alert_counts,
            'total_alerts': len(self.alert_history),
            'module_statistics': module_stats,
            'enabled_modules': self.enable_modules
        }
    
    def reset_statistics(self):
        """Reset all statistics and history."""
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_times.clear()
        self.concentration_history.clear()
        self.alert_history.clear()
        
        # Reset module statistics
        for module in self.modules.values():
            if hasattr(module, 'reset_statistics'):
                module.reset_statistics()
        
        logger.info("All statistics reset")
    
    def start_realtime_analysis(self, camera_id: int = 0):
        """Start real-time analysis from camera."""
        if self.is_running:
            logger.warning("Real-time analysis already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._realtime_processing_loop,
            args=(camera_id,),
            daemon=True
        )
        self.processing_thread.start()
        logger.info(f"Started real-time analysis from camera {camera_id}")
    
    def stop_realtime_analysis(self):
        """Stop real-time analysis."""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Stopped real-time analysis")
    
    def _realtime_processing_loop(self, camera_id: int):
        """Main processing loop for real-time analysis."""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.max_fps)
        
        frame_interval = 1.0 / self.max_fps
        last_process_time = 0
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                current_time = time.time()
                
                # Limit processing rate
                if current_time - last_process_time < frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue
                
                last_process_time = current_time
                
                # Analyze frame
                results = self.analyze_frame(frame)
                
                # Put results in queue for external access
                try:
                    self.result_queue.put_nowait(results)
                except queue.Full:
                    # Remove oldest result if queue is full
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(results)
                    except queue.Empty:
                        pass
                
        except Exception as e:
            logger.error(f"Error in real-time processing loop: {e}")
        finally:
            cap.release()
            logger.info("Camera released")
    
    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """Get the latest analysis results."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_realtime_analysis()
        
        # Cleanup modules
        for module in self.modules.values():
            if hasattr(module, 'cleanup'):
                module.cleanup()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Cleanup feedback system
        if hasattr(self, 'feedback_system'):
            self.feedback_system.cleanup()
        
        # Cleanup data logger
        if hasattr(self, 'data_logger'):
            self.data_logger.cleanup()
        
        logger.info("ConcentrationAnalyzer cleanup completed")
