"""
Drowsiness Analysis Module

Analyzes blink patterns, eye closure duration, and other indicators
to detect drowsiness and fatigue levels. Implements PERCLOS and other
standard drowsiness metrics.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass
import logging

from .ear_calculator import calculate_eye_aspect_ratio, calculate_mar


@dataclass
class DrowsinessMetrics:
    """Container for drowsiness analysis metrics."""
    perclos: float  # Percentage of eyelid closure
    blink_rate: float  # Blinks per minute
    avg_blink_duration: float  # Average blink duration in seconds
    longest_closure: float  # Longest eye closure duration
    eye_closure_velocity: float  # Speed of eye closure
    mar_score: float  # Mouth aspect ratio (yawning indicator)
    fatigue_score: float  # Overall fatigue score (0-1)
    alert_level: str  # Alert level: 'alert', 'drowsy', 'very_drowsy', 'critical'


class DrowsinessAnalyzer:
    """
    Comprehensive drowsiness analysis system that monitors multiple
    physiological indicators of fatigue and sleepiness.
    """
    
    def __init__(self, perclos_threshold: float = 0.8, 
                 drowsy_blink_rate_low: float = 5.0,
                 drowsy_blink_rate_high: float = 25.0,
                 long_closure_threshold: float = 2.0,
                 analysis_window: float = 60.0):
        """
        Initialize drowsiness analyzer.
        
        Args:
            perclos_threshold: EAR threshold for PERCLOS calculation
            drowsy_blink_rate_low: Lower bound for normal blink rate (blinks/min)
            drowsy_blink_rate_high: Upper bound for normal blink rate (blinks/min)
            long_closure_threshold: Threshold for long eye closure (seconds)
            analysis_window: Time window for analysis (seconds)
        """
        self.perclos_threshold = perclos_threshold
        self.drowsy_blink_rate_low = drowsy_blink_rate_low
        self.drowsy_blink_rate_high = drowsy_blink_rate_high
        self.long_closure_threshold = long_closure_threshold
        self.analysis_window = analysis_window
        
        # Data storage
        self.ear_history = deque(maxlen=int(analysis_window * 30))  # Assuming 30 FPS
        self.mar_history = deque(maxlen=int(analysis_window * 30))
        self.timestamps = deque(maxlen=int(analysis_window * 30))
        self.blink_events = deque(maxlen=1000)  # Store blink events
        self.closure_events = deque(maxlen=100)  # Store eye closure events
        
        # State tracking
        self.current_closure_start = None
        self.is_eyes_closed = False
        self.consecutive_low_ear_count = 0
        self.last_blink_time = 0
        
        # Analysis results
        self.current_metrics = None
        
        self.logger = logging.getLogger(__name__)
    
    def add_measurement(self, ear_value: float, mar_value: Optional[float] = None, 
                       timestamp: Optional[float] = None) -> None:
        """
        Add new measurement for analysis.
        
        Args:
            ear_value: Eye aspect ratio value
            mar_value: Mouth aspect ratio value (optional)
            timestamp: Timestamp of measurement (current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Store measurements
        self.ear_history.append(ear_value)
        self.timestamps.append(timestamp)
        
        if mar_value is not None:
            self.mar_history.append(mar_value)
        
        # Track eye closure state
        self._track_eye_closure(ear_value, timestamp)
        
        # Detect blink events
        self._detect_blink_event(ear_value, timestamp)
    
    def _track_eye_closure(self, ear_value: float, timestamp: float) -> None:
        """Track eye closure events for PERCLOS calculation."""
        eye_closed = ear_value < self.perclos_threshold
        
        if eye_closed and not self.is_eyes_closed:
            # Eyes just closed
            self.current_closure_start = timestamp
            self.is_eyes_closed = True
            self.consecutive_low_ear_count = 1
        elif eye_closed and self.is_eyes_closed:
            # Eyes still closed
            self.consecutive_low_ear_count += 1
        elif not eye_closed and self.is_eyes_closed:
            # Eyes just opened
            if self.current_closure_start is not None:
                closure_duration = timestamp - self.current_closure_start
                self.closure_events.append({
                    'start_time': self.current_closure_start,
                    'end_time': timestamp,
                    'duration': closure_duration,
                    'min_ear': min(list(self.ear_history)[-self.consecutive_low_ear_count:])
                })
            
            self.is_eyes_closed = False
            self.current_closure_start = None
            self.consecutive_low_ear_count = 0
    
    def _detect_blink_event(self, ear_value: float, timestamp: float) -> None:
        """Detect and record blink events."""
        # Simple blink detection based on EAR crossing threshold
        if (ear_value < self.perclos_threshold and 
            timestamp - self.last_blink_time > 0.1):  # Minimum 100ms between blinks
            
            self.blink_events.append({
                'timestamp': timestamp,
                'ear_value': ear_value,
                'time_since_last': timestamp - self.last_blink_time
            })
            
            self.last_blink_time = timestamp
    
    def calculate_perclos(self, time_window: Optional[float] = None) -> float:
        """
        Calculate PERCLOS (Percentage of Eyelid Closure) over time window.
        
        Args:
            time_window: Time window in seconds (uses default if None)
            
        Returns:
            PERCLOS value (0-1)
        """
        if time_window is None:
            time_window = self.analysis_window
        
        if len(self.ear_history) == 0 or len(self.timestamps) == 0:
            return 0.0
        
        current_time = time.time()
        
        # Get measurements within time window
        recent_indices = [i for i, t in enumerate(self.timestamps) 
                         if current_time - t <= time_window]
        
        if not recent_indices:
            return 0.0
        
        recent_ear_values = [self.ear_history[i] for i in recent_indices]
        
        # Calculate percentage of time eyes were closed
        closed_count = sum(1 for ear in recent_ear_values if ear < self.perclos_threshold)
        total_count = len(recent_ear_values)
        
        perclos = closed_count / total_count if total_count > 0 else 0.0
        
        return perclos
    
    def calculate_blink_rate(self, time_window: Optional[float] = None) -> float:
        """
        Calculate blink rate (blinks per minute).
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Blink rate in blinks per minute
        """
        if time_window is None:
            time_window = self.analysis_window
        
        current_time = time.time()
        
        # Count recent blinks
        recent_blinks = [b for b in self.blink_events 
                        if current_time - b['timestamp'] <= time_window]
        
        if not recent_blinks:
            return 0.0
        
        # Convert to blinks per minute
        blink_rate = (len(recent_blinks) / time_window) * 60.0
        
        return blink_rate
    
    def calculate_avg_blink_duration(self, time_window: Optional[float] = None) -> float:
        """
        Calculate average blink duration.
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Average blink duration in seconds
        """
        if time_window is None:
            time_window = self.analysis_window
        
        current_time = time.time()
        
        # Get recent closure events
        recent_closures = [c for c in self.closure_events 
                          if current_time - c['end_time'] <= time_window]
        
        if not recent_closures:
            return 0.0
        
        # Filter for blink-like closures (short duration)
        blink_closures = [c for c in recent_closures if c['duration'] < 0.5]
        
        if not blink_closures:
            return 0.0
        
        avg_duration = sum(c['duration'] for c in blink_closures) / len(blink_closures)
        
        return avg_duration
    
    def get_longest_closure(self, time_window: Optional[float] = None) -> float:
        """
        Get the longest eye closure duration in the time window.
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Longest closure duration in seconds
        """
        if time_window is None:
            time_window = self.analysis_window
        
        current_time = time.time()
        
        # Get recent closure events
        recent_closures = [c for c in self.closure_events 
                          if current_time - c['end_time'] <= time_window]
        
        if not recent_closures:
            return 0.0
        
        return max(c['duration'] for c in recent_closures)
    
    def calculate_eye_closure_velocity(self) -> float:
        """
        Calculate the velocity of eye closure (rate of EAR decrease).
        
        Returns:
            Eye closure velocity (EAR units per second)
        """
        if len(self.ear_history) < 10:
            return 0.0
        
        # Look at recent EAR values and timestamps
        recent_ear = list(self.ear_history)[-10:]
        recent_times = list(self.timestamps)[-10:]
        
        # Calculate rate of change
        velocities = []
        for i in range(1, len(recent_ear)):
            dt = recent_times[i] - recent_times[i-1]
            if dt > 0:
                dEAR = recent_ear[i] - recent_ear[i-1]
                velocity = dEAR / dt
                velocities.append(velocity)
        
        if not velocities:
            return 0.0
        
        # Return average velocity (negative values indicate closing)
        return sum(velocities) / len(velocities)
    
    def calculate_mar_score(self, time_window: Optional[float] = None) -> float:
        """
        Calculate average Mouth Aspect Ratio (yawning indicator).
        
        Args:
            time_window: Time window in seconds
            
        Returns:
            Average MAR score
        """
        if not self.mar_history:
            return 0.0
        
        if time_window is None:
            time_window = self.analysis_window
        
        current_time = time.time()
        
        # Get recent MAR values
        recent_indices = [i for i, t in enumerate(self.timestamps) 
                         if current_time - t <= time_window and i < len(self.mar_history)]
        
        if not recent_indices:
            return 0.0
        
        recent_mar_values = [self.mar_history[i] for i in recent_indices]
        
        return sum(recent_mar_values) / len(recent_mar_values)
    
    def calculate_fatigue_score(self) -> float:
        """
        Calculate overall fatigue score based on multiple indicators.
        
        Returns:
            Fatigue score (0-1, where 1 is most fatigued)
        """
        # Get individual metrics
        perclos = self.calculate_perclos()
        blink_rate = self.calculate_blink_rate()
        longest_closure = self.get_longest_closure()
        mar_score = self.calculate_mar_score()
        
        # Weighted combination of factors
        fatigue_score = 0.0
        
        # PERCLOS contribution (0.4 weight)
        if perclos > 0.15:  # Normal PERCLOS is typically < 15%
            fatigue_score += 0.4 * min(1.0, (perclos - 0.15) / 0.35)
        
        # Blink rate contribution (0.2 weight)
        if blink_rate < self.drowsy_blink_rate_low or blink_rate > self.drowsy_blink_rate_high:
            deviation = min(abs(blink_rate - self.drowsy_blink_rate_low),
                          abs(blink_rate - self.drowsy_blink_rate_high))
            fatigue_score += 0.2 * min(1.0, deviation / 10.0)
        
        # Long closure contribution (0.3 weight)
        if longest_closure > 1.0:
            fatigue_score += 0.3 * min(1.0, (longest_closure - 1.0) / 3.0)
        
        # MAR contribution (0.1 weight) - yawning indicator
        if mar_score > 0.5:  # Elevated MAR suggests yawning
            fatigue_score += 0.1 * min(1.0, (mar_score - 0.5) / 0.3)
        
        return min(1.0, fatigue_score)
    
    def determine_alert_level(self, fatigue_score: float) -> str:
        """
        Determine alert level based on fatigue score.
        
        Args:
            fatigue_score: Fatigue score (0-1)
            
        Returns:
            Alert level string
        """
        if fatigue_score < 0.2:
            return 'alert'
        elif fatigue_score < 0.4:
            return 'drowsy'
        elif fatigue_score < 0.7:
            return 'very_drowsy'
        else:
            return 'critical'
    
    def get_current_metrics(self) -> DrowsinessMetrics:
        """
        Get current drowsiness analysis metrics.
        
        Returns:
            DrowsinessMetrics object with all current metrics
        """
        perclos = self.calculate_perclos()
        blink_rate = self.calculate_blink_rate()
        avg_blink_duration = self.calculate_avg_blink_duration()
        longest_closure = self.get_longest_closure()
        eye_closure_velocity = self.calculate_eye_closure_velocity()
        mar_score = self.calculate_mar_score()
        fatigue_score = self.calculate_fatigue_score()
        alert_level = self.determine_alert_level(fatigue_score)
        
        metrics = DrowsinessMetrics(
            perclos=perclos,
            blink_rate=blink_rate,
            avg_blink_duration=avg_blink_duration,
            longest_closure=longest_closure,
            eye_closure_velocity=eye_closure_velocity,
            mar_score=mar_score,
            fatigue_score=fatigue_score,
            alert_level=alert_level
        )
        
        self.current_metrics = metrics
        return metrics
    
    def get_trend_analysis(self, lookback_periods: int = 5) -> Dict[str, str]:
        """
        Analyze trends in drowsiness metrics over multiple time periods.
        
        Args:
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary with trend analysis for each metric
        """
        if lookback_periods < 2:
            return {}
        
        period_length = self.analysis_window / lookback_periods
        trends = {}
        
        # Analyze PERCLOS trend
        perclos_values = []
        for i in range(lookback_periods):
            start_time = time.time() - (i + 1) * period_length
            end_time = time.time() - i * period_length
            
            # Calculate PERCLOS for this period
            period_indices = [j for j, t in enumerate(self.timestamps) 
                            if start_time <= t <= end_time]
            
            if period_indices:
                period_ear_values = [self.ear_history[j] for j in period_indices]
                closed_count = sum(1 for ear in period_ear_values if ear < self.perclos_threshold)
                period_perclos = closed_count / len(period_ear_values)
                perclos_values.append(period_perclos)
        
        if len(perclos_values) >= 2:
            if perclos_values[0] > perclos_values[-1] * 1.2:
                trends['perclos'] = 'increasing'
            elif perclos_values[0] < perclos_values[-1] * 0.8:
                trends['perclos'] = 'decreasing'
            else:
                trends['perclos'] = 'stable'
        
        return trends
    
    def reset_analysis(self) -> None:
        """Reset all analysis data and state."""
        self.ear_history.clear()
        self.mar_history.clear()
        self.timestamps.clear()
        self.blink_events.clear()
        self.closure_events.clear()
        
        self.current_closure_start = None
        self.is_eyes_closed = False
        self.consecutive_low_ear_count = 0
        self.last_blink_time = 0
        self.current_metrics = None
    
    def export_analysis_data(self) -> Dict[str, Any]:
        """
        Export analysis data for external processing or visualization.
        
        Returns:
            Dictionary containing all analysis data
        """
        return {
            'ear_history': list(self.ear_history),
            'mar_history': list(self.mar_history),
            'timestamps': list(self.timestamps),
            'blink_events': list(self.blink_events),
            'closure_events': list(self.closure_events),
            'current_metrics': self.current_metrics.__dict__ if self.current_metrics else None,
            'config': {
                'perclos_threshold': self.perclos_threshold,
                'drowsy_blink_rate_low': self.drowsy_blink_rate_low,
                'drowsy_blink_rate_high': self.drowsy_blink_rate_high,
                'long_closure_threshold': self.long_closure_threshold,
                'analysis_window': self.analysis_window
            }
        }
    
    def generate_alert_message(self, metrics: DrowsinessMetrics) -> str:
        """
        Generate human-readable alert message based on metrics.
        
        Args:
            metrics: Current drowsiness metrics
            
        Returns:
            Alert message string
        """
        if metrics.alert_level == 'alert':
            return "Driver appears alert and focused."
        elif metrics.alert_level == 'drowsy':
            reasons = []
            if metrics.perclos > 0.15:
                reasons.append(f"elevated eye closure ({metrics.perclos:.1%})")
            if metrics.blink_rate < self.drowsy_blink_rate_low:
                reasons.append(f"low blink rate ({metrics.blink_rate:.1f}/min)")
            
            if reasons:
                return f"Mild drowsiness detected: {', '.join(reasons)}."
            else:
                return "Mild drowsiness detected."
        
        elif metrics.alert_level == 'very_drowsy':
            return f"Significant drowsiness detected. PERCLOS: {metrics.perclos:.1%}, " \
                   f"longest closure: {metrics.longest_closure:.1f}s. Consider taking a break."
        
        else:  # critical
            return f"CRITICAL: Severe drowsiness detected! " \
                   f"PERCLOS: {metrics.perclos:.1%}, longest closure: {metrics.longest_closure:.1f}s. " \
                   f"STOP DRIVING IMMEDIATELY."


def create_drowsiness_analyzer(config: Dict[str, Any]) -> DrowsinessAnalyzer:
    """
    Factory function to create drowsiness analyzer with configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured DrowsinessAnalyzer instance
    """
    return DrowsinessAnalyzer(
        perclos_threshold=config.get('perclos_threshold', 0.8),
        drowsy_blink_rate_low=config.get('drowsy_blink_rate_low', 5.0),
        drowsy_blink_rate_high=config.get('drowsy_blink_rate_high', 25.0),
        long_closure_threshold=config.get('long_closure_threshold', 2.0),
        analysis_window=config.get('analysis_window', 60.0)
    )


