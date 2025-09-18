"""
GUI Interface for Concentration Analysis

Simple GUI interface using OpenCV's built-in functions.
For more advanced GUI, consider using PyQt5 or tkinter.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Callable
import time


class GUIInterface:
    """
    Simple GUI interface for displaying concentration analysis results.
    """
    
    def __init__(self, window_name: str = "Concentration Analysis",
                 window_size: tuple = (1200, 800)):
        """
        Initialize GUI interface.
        
        Args:
            window_name: Main window name
            window_size: Window size (width, height)
        """
        self.window_name = window_name
        self.window_size = window_size
        
        # Display settings
        self.show_fps = True
        self.show_metrics = True
        self.show_recommendations = True
        
        # Colors
        self.colors = {
            'high': (0, 255, 0),      # Green
            'medium': (0, 255, 255),  # Yellow
            'low': (0, 165, 255),     # Orange
            'distracted': (0, 0, 255) # Red
        }
        
        # GUI state
        self.is_initialized = False
        self.last_fps_update = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        
        # Callbacks
        self.key_callbacks = {}
    
    def initialize(self) -> bool:
        """
        Initialize GUI components.
        
        Returns:
            True if successful
        """
        try:
            # Create main window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"GUI initialization failed: {e}")
            return False
    
    def display_frame(self, frame: np.ndarray, 
                     analysis_results: Optional[Dict[str, Any]] = None) -> int:
        """
        Display frame with analysis results.
        
        Args:
            frame: Input video frame
            analysis_results: Analysis results to overlay
            
        Returns:
            Key code pressed (or -1 if none)
        """
        if not self.is_initialized:
            if not self.initialize():
                return -1
        
        # Create display frame
        display_frame = frame.copy()
        
        # Add analysis overlays
        if analysis_results:
            display_frame = self._add_analysis_overlay(display_frame, analysis_results)
        
        # Add FPS counter
        if self.show_fps:
            display_frame = self._add_fps_counter(display_frame)
        
        # Display frame
        cv2.imshow(self.window_name, display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Process callbacks
        if key != 255 and key in self.key_callbacks:
            self.key_callbacks[key]()
        
        return key
    
    def _add_analysis_overlay(self, frame: np.ndarray, 
                            results: Dict[str, Any]) -> np.ndarray:
        """Add analysis results overlay to frame."""
        overlay_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Extract metrics if available
        if 'overall_score' in results:
            score = results['overall_score']
            attention_level = results.get('attention_level', 'unknown')
            
            # Draw concentration bar
            self._draw_concentration_bar(overlay_frame, score, attention_level)
        
        # Draw component scores
        if self.show_metrics:
            self._draw_component_scores(overlay_frame, results)
        
        # Draw recommendations
        if self.show_recommendations and 'recommendations' in results:
            self._draw_recommendations(overlay_frame, results['recommendations'])
        
        return overlay_frame
    
    def _draw_concentration_bar(self, frame: np.ndarray, score: float, level: str):
        """Draw concentration score bar."""
        h, w = frame.shape[:2]
        
        # Bar dimensions
        bar_width = 300
        bar_height = 30
        bar_x = w - bar_width - 20
        bar_y = 20
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (64, 64, 64), -1)
        
        # Score bar
        fill_width = int(bar_width * score)
        color = self.colors.get(level, (128, 128, 128))
        
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Text
        cv2.putText(frame, f"Concentration: {score:.2f}", 
                   (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Level: {level.upper()}", 
                   (bar_x, bar_y + bar_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_component_scores(self, frame: np.ndarray, results: Dict[str, Any]):
        """Draw individual component scores."""
        y_start = 80
        x_pos = 20
        
        components = [
            ('Gaze Focus', results.get('gaze_focus', 0.0)),
            ('Alertness', results.get('alertness', 0.0)),
            ('Head Stability', results.get('head_stability', 0.0)),
            ('Engagement', results.get('engagement', 0.0))
        ]
        
        for i, (name, score) in enumerate(components):
            y_pos = y_start + i * 30
            
            # Score bar
            bar_width = 150
            bar_height = 15
            fill_width = int(bar_width * score)
            
            # Background
            cv2.rectangle(frame, (x_pos + 120, y_pos - 10), 
                         (x_pos + 120 + bar_width, y_pos + 5), (64, 64, 64), -1)
            
            # Fill
            if score > 0.7:
                color = (0, 255, 0)
            elif score > 0.4:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            
            cv2.rectangle(frame, (x_pos + 120, y_pos - 10), 
                         (x_pos + 120 + fill_width, y_pos + 5), color, -1)
            
            # Text
            cv2.putText(frame, f"{name}:", (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"{score:.2f}", (x_pos + 280, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_recommendations(self, frame: np.ndarray, recommendations: list):
        """Draw recommendations text."""
        if not recommendations:
            return
        
        h, w = frame.shape[:2]
        y_start = h - 100
        
        cv2.putText(frame, "Recommendations:", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        for i, rec in enumerate(recommendations[:3]):  # Show max 3 recommendations
            y_pos = y_start + 25 + i * 20
            cv2.putText(frame, f"• {rec}", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _add_fps_counter(self, frame: np.ndarray) -> np.ndarray:
        """Add FPS counter to frame."""
        # Update FPS calculation
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_update = current_time
        else:
            self.fps_counter += 1
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def add_key_callback(self, key: int, callback: Callable[[], None]):
        """
        Add keyboard callback.
        
        Args:
            key: Key code (e.g., ord('q'))
            callback: Function to call when key is pressed
        """
        self.key_callbacks[key] = callback
    
    def show_help_dialog(self):
        """Show help dialog with keyboard shortcuts."""
        help_text = [
            "Keyboard Shortcuts:",
            "q - Quit application",
            "s - Save current session",
            "r - Reset analysis",
            "h - Show/hide help",
            "f - Toggle FPS display",
            "m - Toggle metrics display"
        ]
        
        # Create help window
        help_img = np.zeros((300, 400, 3), dtype=np.uint8)
        
        for i, line in enumerate(help_text):
            cv2.putText(help_img, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Help", help_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Help")
    
    def cleanup(self):
        """Clean up GUI resources."""
        cv2.destroyAllWindows()
        self.is_initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

