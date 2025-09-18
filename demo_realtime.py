#!/usr/bin/env python3
"""
Real-time Concentration Analysis Demo

This script demonstrates the multi-modal concentration analysis system
in real-time using webcam input.
"""

import cv2
import numpy as np
import time
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from integration.concentration_analyzer import ConcentrationAnalyzer
except ImportError:
    # Fallback for development
    import os
    os.chdir(str(Path(__file__).parent / 'src'))
    from integration.concentration_analyzer import ConcentrationAnalyzer


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('concentration_analysis.log')
        ]
    )


def create_visualization(frame: np.ndarray, results: dict) -> np.ndarray:
    """Create comprehensive visualization of analysis results."""
    vis_frame = frame.copy()
    h, w = vis_frame.shape[:2]
    
    # Main info panel
    panel_height = 200
    panel_width = 400
    
    # Semi-transparent background
    overlay = vis_frame.copy()
    cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
    vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
    
    # Border
    concentration_score = results.get('concentration_score', 0.5)
    if concentration_score >= 0.7:
        border_color = (0, 255, 0)  # Green
    elif concentration_score >= 0.4:
        border_color = (0, 255, 255)  # Yellow
    else:
        border_color = (0, 0, 255)  # Red
    
    cv2.rectangle(vis_frame, (10, 10), (panel_width, panel_height), border_color, 3)
    
    # Title
    cv2.putText(vis_frame, "Concentration Analysis", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    y_offset = 60
    line_height = 25
    
    # Concentration score
    score_text = f"Concentration: {concentration_score:.2f}"
    cv2.putText(vis_frame, score_text, (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += line_height
    
    # Attention level
    attention_level = results.get('attention_level', 'unknown')
    attention_text = f"Attention: {attention_level.replace('_', ' ').title()}"
    cv2.putText(vis_frame, attention_text, (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += line_height
    
    # Module results
    modules_info = []
    
    # Gaze
    gaze_results = results.get('gaze')
    if gaze_results and gaze_results.get('valid'):
        gaze_angles = gaze_results.get('gaze_angles', (0, 0))
        modules_info.append(f"Gaze: ({gaze_angles[0]:.1f}°, {gaze_angles[1]:.1f}°)")
    else:
        modules_info.append("Gaze: Not available")
    
    # Head pose
    pose_results = results.get('pose')
    if pose_results and pose_results.get('valid'):
        pitch, yaw, roll = pose_results.get('pitch', 0), pose_results.get('yaw', 0), pose_results.get('roll', 0)
        modules_info.append(f"Pose: P{pitch:.1f}° Y{yaw:.1f}° R{roll:.1f}°")
    else:
        modules_info.append("Pose: Not available")
    
    # Drowsiness
    drowsiness_results = results.get('drowsiness')
    if drowsiness_results:
        drowsiness_level = drowsiness_results.get('drowsiness_level')
        if drowsiness_level and hasattr(drowsiness_level, 'name'):
            modules_info.append(f"Alertness: {drowsiness_level.name}")
        else:
            modules_info.append("Alertness: Unknown")
    else:
        modules_info.append("Alertness: Not available")
    
    # Engagement
    engagement_results = results.get('engagement')
    if engagement_results:
        engagement_level = engagement_results.get('engagement_level')
        if engagement_level and hasattr(engagement_level, 'name'):
            modules_info.append(f"Engagement: {engagement_level.name}")
        else:
            modules_info.append("Engagement: Unknown")
    else:
        modules_info.append("Engagement: Not available")
    
    # Display module info
    for info in modules_info:
        cv2.putText(vis_frame, info, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
    
    # Alerts
    alerts = results.get('alerts', [])
    if alerts:
        alert_y = panel_height + 30
        cv2.putText(vis_frame, "ALERTS:", (20, alert_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        for i, alert in enumerate(alerts[:3]):  # Show max 3 alerts
            alert_text = f"• {alert.get('message', 'Unknown alert')}"
            cv2.putText(vis_frame, alert_text, (20, alert_y + 25 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
    
    # Performance info
    processing_time = results.get('processing_time', 0)
    fps_text = f"Processing: {processing_time*1000:.1f}ms"
    cv2.putText(vis_frame, fps_text, (w - 200, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Concentration score bar
    bar_x = w - 60
    bar_y = 60
    bar_height = 200
    bar_width = 30
    
    # Background
    cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                 (50, 50, 50), -1)
    
    # Fill based on concentration score
    fill_height = int(concentration_score * bar_height)
    fill_y = bar_y + bar_height - fill_height
    
    cv2.rectangle(vis_frame, (bar_x, fill_y), (bar_x + bar_width, bar_y + bar_height), 
                 border_color, -1)
    
    # Bar label
    cv2.putText(vis_frame, "Focus", (bar_x - 10, bar_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_frame


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Real-time Concentration Analysis Demo')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--fps', type=float, default=15.0, help='Target FPS (default: 15)')
    parser.add_argument('--modules', nargs='+', 
                       choices=['gaze', 'blink', 'drowsiness', 'pose', 'engagement'],
                       default=['blink', 'drowsiness', 'pose', 'engagement'],
                       help='Modules to enable')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--save-stats', type=str, help='Save statistics to file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting concentration analysis demo with modules: {args.modules}")
    
    # Configuration
    config = {
        'gaze': {
            'model_type': 'resnet',
            'model_path': None  # Will use default/untrained model
        },
        'blink': {
            'method': 'ear',  # Use EAR method as it doesn't require trained models
            'model_path': None
        },
        'drowsiness': {
            'model_path': None
        },
        'pose': {
            'model_type': 'resnet',
            'model_path': None
        },
        'engagement': {
            'model_type': 'cnn',
            'model_path': None
        }
    }
    
    try:
        # Initialize analyzer
        logger.info("Initializing concentration analyzer...")
        analyzer = ConcentrationAnalyzer(
            config=config,
            enable_modules=args.modules,
            max_fps=args.fps,
            enable_feedback=False,  # Disable for demo
            enable_logging=False    # Disable for demo
        )
        
        # Initialize camera
        logger.info(f"Opening camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {args.camera}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        
        logger.info("Starting real-time analysis. Press 'q' to quit, 's' to save stats.")
        
        # Main processing loop
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                continue
            
            # Analyze frame
            results = analyzer.analyze_frame(frame)
            frame_count += 1
            
            # Create visualization
            if not args.no_display:
                vis_frame = create_visualization(frame, results)
                
                cv2.imshow('Concentration Analysis', vis_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and args.save_stats:
                    stats = analyzer.get_statistics()
                    import json
                    with open(args.save_stats, 'w') as f:
                        json.dump(stats, f, indent=2, default=str)
                    logger.info(f"Statistics saved to {args.save_stats}")
                elif key == ord('r'):
                    analyzer.reset_statistics()
                    logger.info("Statistics reset")
            
            # Print periodic statistics
            if frame_count % 150 == 0:  # Every ~10 seconds at 15fps
                elapsed_time = time.time() - start_time
                actual_fps = frame_count / elapsed_time
                
                stats = analyzer.get_statistics()
                logger.info(f"Frame {frame_count}: "
                           f"FPS={actual_fps:.1f}, "
                           f"Avg Concentration={stats['average_concentration']:.2f}, "
                           f"Alerts={stats['total_alerts']}")
        
        # Final statistics
        elapsed_time = time.time() - start_time
        final_stats = analyzer.get_statistics()
        
        logger.info("=== Final Statistics ===")
        logger.info(f"Session Duration: {elapsed_time:.1f}s")
        logger.info(f"Frames Processed: {frame_count}")
        logger.info(f"Average FPS: {frame_count/elapsed_time:.1f}")
        logger.info(f"Average Concentration: {final_stats['average_concentration']:.2f}")
        logger.info(f"Total Alerts: {final_stats['total_alerts']}")
        
        if args.save_stats:
            import json
            with open(args.save_stats, 'w') as f:
                json.dump(final_stats, f, indent=2, default=str)
            logger.info(f"Final statistics saved to {args.save_stats}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in demo: {e}", exc_info=True)
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        if 'analyzer' in locals():
            analyzer.cleanup()
        cv2.destroyAllWindows()
        logger.info("Demo ended")


if __name__ == '__main__':
    main()
