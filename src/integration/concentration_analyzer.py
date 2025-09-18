"""
Main Concentration Analysis System

Integrates all modalities to provide comprehensive concentration assessment.
Combines eye gaze, blink detection, head pose, and engagement recognition
to generate real-time concentration scores and feedback.
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import logging

# Import all modules
import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

try:
    from modules.eye_gaze import GazeEstimator
    from modules.blink_detection import BlinkDetector, DrowsinessAnalyzer
    from modules.head_pose import HeadPoseEstimator
    # from modules.engagement import EngagementRecognizer  # Will be implemented
    
    from integration.fusion_engine import MultiModalFusion
    from integration.scoring import ConcentrationScorer
except ImportError:
    # Fallback for relative imports
    from ..modules.eye_gaze import GazeEstimator
    from ..modules.blink_detection import BlinkDetector, DrowsinessAnalyzer
    from ..modules.head_pose import HeadPoseEstimator
    # from ..modules.engagement import EngagementRecognizer  # Will be implemented
    
    from .fusion_engine import MultiModalFusion
    from .scoring import ConcentrationScorer


@dataclass
class ConcentrationMetrics:
    """Container for concentration analysis results."""
    overall_score: float  # 0-1 concentration score
    attention_level: str  # 'high', 'medium', 'low', 'distracted'
    gaze_focus: float  # Gaze stability score
    alertness: float  # Based on blink patterns
    head_stability: float  # Head pose stability
    engagement: float  # Facial engagement score
    confidence: float  # Overall confidence in assessment
    recommendations: List[str]  # Actionable recommendations


class ConcentrationAnalyzer:
    """
    Main system that orchestrates all concentration analysis components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the concentration analyzer.
        
        Args:
            config: Configuration dictionary for all components
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize individual components
        self._initialize_components()
        
        # Initialize fusion and scoring engines
        self.fusion_engine = MultiModalFusion(config.get('fusion', {}))
        self.concentration_scorer = ConcentrationScorer(config.get('scoring', {}))
        
        # Analysis state
        self.analysis_history = deque(maxlen=300)  # 10 seconds at 30 FPS
        self.current_metrics = None
        self.frame_count = 0
        self.start_time = time.time()
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Track eye-closure state across frames for smoothing resets
        self.previous_eyes_closed = False
        
    def _initialize_components(self):
        """Initialize all analysis components."""
        try:
            # Eye gaze estimator
            gaze_config = self.config.get('gaze', {})
            self.gaze_estimator = GazeEstimator(
                model_path=gaze_config.get('model_path', 'models/gaze_model.pth'),
                device=gaze_config.get('device', 'cpu')
            )
            self.logger.info("Gaze estimator initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize gaze estimator: {e}")
            self.gaze_estimator = None
        
        try:
            # Blink detector
            blink_config = self.config.get('blink', {})
            self.blink_detector = BlinkDetector(
                method=blink_config.get('method', 'ear'),
                ear_threshold=blink_config.get('ear_threshold', 0.25)
            )
            self.logger.info("Blink detector initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize blink detector: {e}")
            self.blink_detector = None
        
        try:
            # Drowsiness analyzer
            drowsiness_config = self.config.get('drowsiness', {})
            self.drowsiness_analyzer = DrowsinessAnalyzer(
                perclos_threshold=drowsiness_config.get('perclos_threshold', 0.8),
                analysis_window=drowsiness_config.get('analysis_window', 60.0)
            )
            self.logger.info("Drowsiness analyzer initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize drowsiness analyzer: {e}")
            self.drowsiness_analyzer = None
        
        try:
            # Head pose estimator
            pose_config = self.config.get('head_pose', {})
            self.head_pose_estimator = HeadPoseEstimator(
                method=pose_config.get('method', 'pnp'),
                model_path=pose_config.get('model_path', None)
            )
            self.logger.info("Head pose estimator initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize head pose estimator: {e}")
            self.head_pose_estimator = None
        
        # TODO: Initialize engagement recognizer when implemented
        self.engagement_recognizer = None
    
    def analyze_frame(self, frame: np.ndarray) -> ConcentrationMetrics:
        """
        Analyze a single frame for concentration indicators.
        
        Args:
            frame: Input video frame
            
        Returns:
            ConcentrationMetrics with analysis results
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Initialize results dictionary
        results = {
            'timestamp': start_time,
            'frame_id': self.frame_count,
            'gaze': None,
            'blink': None,
            'drowsiness': None,
            'head_pose': None,
            'engagement': None
        }
        
        # Detect facial landmarks (shared across components)
        landmarks = None
        if self.head_pose_estimator:
            landmarks = self.head_pose_estimator.detect_landmarks(frame)
        
        # Run individual analyses
        if self.gaze_estimator:
            try:
                gaze_result = self.gaze_estimator.estimate_gaze(frame)
                results['gaze'] = {
                    'yaw': gaze_result[0],
                    'pitch': gaze_result[1],
                    'confidence': gaze_result[2]
                }
                # Expose face/eyes detection from gaze estimator
                if hasattr(self.gaze_estimator, 'last_detection') and isinstance(self.gaze_estimator.last_detection, dict):
                    results['gaze_detection'] = self.gaze_estimator.last_detection
            except Exception as e:
                self.logger.error(f"Gaze estimation failed: {e}")
        
        if self.blink_detector:
            try:
                blink_result = self.blink_detector.detect_blink(frame)
                results['blink'] = blink_result
                
                # Update drowsiness analyzer
                if self.drowsiness_analyzer and 'ear_value' in blink_result:
                    self.drowsiness_analyzer.add_measurement(
                        blink_result['ear_value'],
                        mar_value=blink_result.get('mar_value')
                    )
                    drowsiness_metrics = self.drowsiness_analyzer.get_current_metrics()
                    results['drowsiness'] = drowsiness_metrics.__dict__
                    # Track current eyes-closed state for stronger penalties
                    eyes_closed_now = bool(self.drowsiness_analyzer.is_eyes_closed)
                    results['eyes_closed'] = eyes_closed_now
                    
                    # Mark recent closure window (to avoid instant full recovery)
                    recent_closure = False
                    try:
                        if (not eyes_closed_now and len(self.drowsiness_analyzer.closure_events) > 0):
                            last_event = self.drowsiness_analyzer.closure_events[-1]
                            recent_closure = (time.time() - last_event.get('end_time', 0)) < 0.75
                    except Exception:
                        recent_closure = False
                    results['recent_closure'] = recent_closure
                    
                    # If eyes just reopened, reset smoothing to allow recovery
                    if self.previous_eyes_closed and not eyes_closed_now:
                        try:
                            if hasattr(self.fusion_engine, 'score_history'):
                                self.fusion_engine.score_history.clear()
                            if hasattr(self.concentration_scorer, 'score_history'):
                                self.concentration_scorer.score_history.clear()
                        except Exception:
                            pass
                    self.previous_eyes_closed = eyes_closed_now
            except Exception as e:
                self.logger.error(f"Blink detection failed: {e}")
        
        if self.head_pose_estimator:
            try:
                pose_result = self.head_pose_estimator.estimate_pose(frame, landmarks)
                results['head_pose'] = pose_result
            except Exception as e:
                self.logger.error(f"Head pose estimation failed: {e}")
        
        # TODO: Add engagement analysis when implemented
        
        # Detect if a person is present in the frame
        person_present = False
        try:
            if results.get('head_pose') and isinstance(results['head_pose'], dict):
                person_present = bool(results['head_pose'].get('success', False))
            if not person_present and results.get('gaze'):
                person_present = results['gaze'].get('confidence', 0.0) >= 0.3
        except Exception:
            person_present = False

        # If no person detected, output low score and reset smoothing histories
        if not person_present:
            try:
                if hasattr(self.fusion_engine, 'score_history'):
                    self.fusion_engine.score_history.clear()
                if hasattr(self.concentration_scorer, 'score_history'):
                    self.concentration_scorer.score_history.clear()
            except Exception:
                pass
            metrics = ConcentrationMetrics(
                overall_score=0.05,
                attention_level='distracted',
                gaze_focus=0.0,
                alertness=0.0,
                head_stability=0.0,
                engagement=0.0,
                confidence=0.1,
                recommendations=["No person detected"]
            )
            # Update history
            self.analysis_history.append(results)
            self.current_metrics = metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            return metrics

        # Fuse results and compute concentration score
        concentration_metrics = self._compute_concentration_metrics(results)
        
        # Update history
        self.analysis_history.append(results)
        self.current_metrics = concentration_metrics
        
        # Track performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return concentration_metrics
    
    def _compute_concentration_metrics(self, results: Dict[str, Any]) -> ConcentrationMetrics:
        """
        Compute overall concentration metrics from individual results.
        
        Args:
            results: Dictionary of analysis results from all components
            
        Returns:
            ConcentrationMetrics object
        """
        # Extract individual scores
        gaze_focus = self._compute_gaze_focus_score(results.get('gaze'))
        alertness = self._compute_alertness_score(results.get('drowsiness'))
        head_stability = self._compute_head_stability_score(results.get('head_pose'))
        engagement = self._compute_engagement_score(results.get('engagement'))
        
        # Determine eyes-closed signal from drowsiness and/or gaze detector fallback
        eyes_closed = bool(results.get('eyes_closed', False))
        recent_closure = bool(results.get('recent_closure', False))
        if not eyes_closed:
            gd = results.get('gaze_detection') or {}
            eyes_det = gd.get('eyes_detected', False)
            face_det = gd.get('face_detected', False)
            # If face is detected but eyes are not for multiple frames, treat as closed
            if face_det and not eyes_det:
                # Maintain a small counter in instance state
                closed_counter = getattr(self, '_no_eye_counter', 0) + 1
                self._no_eye_counter = closed_counter
                if closed_counter >= 5:  # ~5 frames ~ 0.15s at 30fps
                    eyes_closed = True
            else:
                self._no_eye_counter = 0

        if eyes_closed:
            gaze_focus = 0.0
            alertness = min(alertness, 0.1)
        elif recent_closure:
            gaze_focus = gaze_focus * 0.4
            alertness = min(alertness, 0.4)

        # Increase fatigue penalty if yawn detected via MAR
        if results.get('blink') and results['blink'].get('yawn_detected'):
            alertness = min(alertness, 0.3)

        # Use fusion engine to combine scores
        fused_score = self.fusion_engine.fuse_scores({
            'gaze_focus': gaze_focus,
            'alertness': alertness,
            'head_stability': head_stability,
            'engagement': engagement
        })

        # Compute overall concentration score
        overall_score = self.concentration_scorer.compute_concentration_score(fused_score)

        # Enforce caps only during active or very recent closures
        if eyes_closed:
            overall_score = min(overall_score, 0.15)
        elif recent_closure:
            overall_score = min(overall_score, 0.4)

        # Determine attention level
        attention_level = self._determine_attention_level(overall_score)

        # Calculate confidence
        confidence = self._calculate_confidence(results)

        # Generate recommendations
        recommendations = self._generate_recommendations(results, overall_score)

        return ConcentrationMetrics(
            overall_score=overall_score,
            attention_level=attention_level,
            gaze_focus=gaze_focus,
            alertness=alertness,
            head_stability=head_stability,
            engagement=engagement,
            confidence=confidence,
            recommendations=recommendations
        )
        
        # Compute overall concentration score
        overall_score = self.concentration_scorer.compute_concentration_score(fused_score)
        
        # Determine attention level
        attention_level = self._determine_attention_level(overall_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, overall_score)
        
        return ConcentrationMetrics(
            overall_score=overall_score,
            attention_level=attention_level,
            gaze_focus=gaze_focus,
            alertness=alertness,
            head_stability=head_stability,
            engagement=engagement,
            confidence=confidence,
            recommendations=recommendations
        )
    
    def _compute_gaze_focus_score(self, gaze_result: Optional[Dict]) -> float:
        """Compute gaze focus score from gaze estimation results."""
        if not gaze_result or gaze_result['confidence'] < 0.3:
            return 0.5  # Neutral score if no reliable gaze data
        
        # Score based on gaze stability and central fixation
        yaw, pitch = gaze_result['yaw'], gaze_result['pitch']
        
        # Central gaze gets higher score
        center_distance = np.sqrt(yaw**2 + pitch**2)
        center_score = max(0, 1.0 - center_distance / 45.0)  # 45 degrees max
        
        # Factor in confidence
        focus_score = center_score * gaze_result['confidence']
        
        return min(1.0, max(0.0, focus_score))
    
    def _compute_alertness_score(self, drowsiness_result: Optional[Dict]) -> float:
        """Compute alertness score from drowsiness analysis."""
        if not drowsiness_result:
            return 0.7  # Default moderate alertness
        
        # Invert fatigue score to get alertness
        fatigue_score = drowsiness_result.get('fatigue_score', 0.0)
        alertness = 1.0 - fatigue_score
        
        return min(1.0, max(0.0, alertness))
    
    def _compute_head_stability_score(self, pose_result: Optional[Dict]) -> float:
        """Compute head stability score from pose estimation."""
        if not pose_result or not pose_result.get('success'):
            return 0.5  # Neutral score if no pose data
        
        # Get pose stability from estimator
        if self.head_pose_estimator:
            stability = self.head_pose_estimator.get_pose_stability()
            # Lower stability values are better (more stable)
            stability_score = max(0, 1.0 - stability / 20.0)  # 20 degrees max deviation
            return min(1.0, stability_score)
        
        return 0.5
    
    def _compute_engagement_score(self, engagement_result: Optional[Dict]) -> float:
        """Compute engagement score from facial expression analysis."""
        if not engagement_result:
            return 0.6  # Default moderate engagement
        
        # TODO: Implement when engagement recognizer is ready
        return 0.6
    
    def _determine_attention_level(self, overall_score: float) -> str:
        """Determine categorical attention level from overall score."""
        if overall_score >= 0.8:
            return 'high'
        elif overall_score >= 0.6:
            return 'medium'
        elif overall_score >= 0.4:
            return 'low'
        else:
            return 'distracted'
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence in the analysis."""
        confidences = []
        
        # Gaze confidence
        if results.get('gaze') and 'confidence' in results['gaze']:
            confidences.append(results['gaze']['confidence'])
        
        # Blink detection confidence
        if results.get('blink') and 'confidence' in results['blink']:
            confidences.append(results['blink']['confidence'])
        
        # Head pose confidence
        if results.get('head_pose') and 'confidence' in results['head_pose']:
            confidences.append(results['head_pose']['confidence'])
        
        if not confidences:
            return 0.5  # Default confidence
        
        # Weighted average of confidences
        return sum(confidences) / len(confidences)
    
    def _generate_recommendations(self, results: Dict[str, Any], 
                                overall_score: float) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Check individual components for specific issues
        if results.get('drowsiness'):
            drowsiness = results['drowsiness']
            if drowsiness.get('alert_level') in ['very_drowsy', 'critical']:
                recommendations.append("Take a break - high drowsiness detected")
            elif drowsiness.get('perclos', 0) > 0.2:
                recommendations.append("Consider taking a short rest")
        
        if results.get('gaze'):
            gaze = results['gaze']
            if gaze.get('confidence', 0) > 0.5:
                yaw, pitch = gaze.get('yaw', 0), gaze.get('pitch', 0)
                if abs(yaw) > 30:
                    recommendations.append("Focus attention toward center")
                if pitch < -20:
                    recommendations.append("Look up - posture may be affecting concentration")
        
        if results.get('head_pose'):
            pose = results['head_pose']
            if pose.get('success') and self.head_pose_estimator:
                stability = self.head_pose_estimator.get_pose_stability()
                if stability > 15:
                    recommendations.append("Try to keep head more stable")
        
        # Overall recommendations
        if overall_score < 0.4:
            recommendations.append("Multiple distraction indicators detected")
        elif overall_score < 0.6:
            recommendations.append("Consider minimizing distractions")
        
        if not recommendations:
            recommendations.append("Concentration levels appear good")
        
        return recommendations[:3]  # Limit to top 3 recommendations
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the entire analysis session."""
        if not self.analysis_history:
            return {}
        
        # Calculate session statistics
        total_time = time.time() - self.start_time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        # Get concentration score history
        concentration_scores = []
        for result in self.analysis_history:
            if hasattr(result, 'overall_score'):
                concentration_scores.append(result.overall_score)
        
        summary = {
            'session_duration': total_time,
            'frames_processed': self.frame_count,
            'avg_processing_time': avg_processing_time,
            'avg_fps': self.frame_count / total_time if total_time > 0 else 0,
            'concentration_stats': {
                'mean': np.mean(concentration_scores) if concentration_scores else 0,
                'std': np.std(concentration_scores) if concentration_scores else 0,
                'min': np.min(concentration_scores) if concentration_scores else 0,
                'max': np.max(concentration_scores) if concentration_scores else 0
            }
        }
        
        return summary
    
    def visualize_analysis(self, frame: np.ndarray, 
                          metrics: ConcentrationMetrics) -> np.ndarray:
        """
        Visualize concentration analysis results on frame.
        
        Args:
            frame: Input frame
            metrics: Analysis results
            
        Returns:
            Frame with analysis visualization
        """
        result_frame = frame.copy()
        
        # Draw concentration score bar
        bar_width = 300
        bar_height = 20
        bar_x = frame.shape[1] - bar_width - 20
        bar_y = 20
        
        # Background bar
        cv2.rectangle(result_frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (64, 64, 64), -1)
        
        # Concentration level bar
        fill_width = int(bar_width * metrics.overall_score)
        if metrics.overall_score >= 0.7:
            color = (0, 255, 0)  # Green
        elif metrics.overall_score >= 0.4:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(result_frame, (bar_x, bar_y), 
                     (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Text labels
        cv2.putText(result_frame, f"Concentration: {metrics.overall_score:.2f}", 
                   (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(result_frame, f"Level: {metrics.attention_level.upper()}", 
                   (bar_x, bar_y + bar_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Component scores
        y_offset = 60
        components = [
            ('Gaze Focus', metrics.gaze_focus),
            ('Alertness', metrics.alertness),
            ('Head Stability', metrics.head_stability),
            ('Engagement', metrics.engagement)
        ]
        
        for name, score in components:
            cv2.putText(result_frame, f"{name}: {score:.2f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Recommendations
        if metrics.recommendations:
            cv2.putText(result_frame, "Recommendations:", 
                       (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 35
            
            for i, rec in enumerate(metrics.recommendations[:2]):  # Show top 2
                cv2.putText(result_frame, f"• {rec}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
        
        return result_frame
    
    def export_session_data(self, filepath: str) -> bool:
        """
        Export session analysis data to file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export successful
        """
        try:
            import json
            
            export_data = {
                'session_summary': self.get_session_summary(),
                'configuration': self.config,
                'analysis_history': list(self.analysis_history)
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Session data exported to {filepath}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to export session data: {e}")
            return False


def create_concentration_analyzer(config_path: str) -> ConcentrationAnalyzer:
    """
    Factory function to create concentration analyzer from config file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConcentrationAnalyzer instance
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return ConcentrationAnalyzer(config)


