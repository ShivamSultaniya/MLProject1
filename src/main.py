"""
Main Entry Point for Multi-Modal Concentration Analysis System

This script provides the main interface for running the concentration analysis system
with real-time webcam input and various analysis modes.
"""

import cv2
import argparse
import yaml
import logging
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from integration.concentration_analyzer import ConcentrationAnalyzer
except ImportError:
    # Fallback for when running from project root
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from integration.concentration_analyzer import ConcentrationAnalyzer


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('concentration_analysis.log')
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using default configuration")
        return get_default_config()
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return get_default_config()


def get_default_config() -> dict:
    """Get default configuration for the system."""
    return {
        'gaze': {
            'model_path': 'models/gaze_model.pth',
            'device': 'cpu',
            'method': 'gazenet'
        },
        'blink': {
            'method': 'ear',
            'ear_threshold': 0.25,
            'consecutive_frames': 3
        },
        'drowsiness': {
            'perclos_threshold': 0.25,
            'analysis_window': 60.0,
            'drowsy_blink_rate_low': 5.0,
            'drowsy_blink_rate_high': 25.0
        },
        'head_pose': {
            'method': 'pnp',
            'model_path': None,
            'smoothing_window': 5
        },
        'fusion': {
            'weights': {
                'gaze_focus': 0.3,
                'alertness': 0.3,
                'head_stability': 0.2,
                'engagement': 0.2
            }
        },
        'scoring': {
            'temporal_smoothing': True,
            'smoothing_window': 10
        },
        'display': {
            'show_visualization': True,
            'show_fps': True,
            'save_video': False,
            'video_output_path': 'output_video.mp4'
        }
    }


def run_real_time_analysis(config: dict, video_source: int = 0, 
                          duration: int = None, output_path: str = None):
    """
    Run real-time concentration analysis.
    
    Args:
        config: System configuration
        video_source: Video source (0 for webcam, or video file path)
        duration: Maximum duration in seconds (None for unlimited)
        output_path: Path to save analysis results
    """
    # Initialize concentration analyzer
    analyzer = ConcentrationAnalyzer(config)
    
    # Setup video capture
    if isinstance(video_source, str):
        cap = cv2.VideoCapture(video_source)
    else:
        cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        logging.error("Error: Could not open video source")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    # Setup video writer if saving output
    video_writer = None
    if config.get('display', {}).get('save_video', False):
        output_video_path = config['display'].get('video_output_path', 'output_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Analysis loop
    start_time = time.time()
    frame_count = 0
    fps_counter = 0
    last_fps_time = time.time()
    
    logging.info("Starting real-time concentration analysis...")
    logging.info("Press 'q' to quit, 's' to save current session data")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from video source")
                break
            
            frame_start_time = time.time()
            
            # Analyze frame
            metrics = analyzer.analyze_frame(frame)
            
            # Visualize results
            if config.get('display', {}).get('show_visualization', True):
                result_frame = analyzer.visualize_analysis(frame, metrics)
                
                # Add FPS counter
                if config.get('display', {}).get('show_fps', True):
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        fps_display = fps_counter
                        fps_counter = 0
                        last_fps_time = current_time
                    else:
                        fps_counter += 1
                    
                    cv2.putText(result_frame, f"FPS: {fps_display if 'fps_display' in locals() else 0}", 
                               (frame_width - 100, frame_height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('Concentration Analysis', result_frame)
                
                # Save frame if recording
                if video_writer:
                    video_writer.write(result_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("Quit requested by user")
                break
            elif key == ord('s'):
                # Save session data
                timestamp = int(time.time())
                save_path = f"session_data_{timestamp}.json"
                if analyzer.export_session_data(save_path):
                    logging.info(f"Session data saved to {save_path}")
                else:
                    logging.error("Failed to save session data")
            elif key == ord('r'):
                # Reset analysis
                analyzer = ConcentrationAnalyzer(config)
                logging.info("Analysis reset")
            
            frame_count += 1
            
            # Check duration limit
            if duration and (time.time() - start_time) >= duration:
                logging.info(f"Duration limit of {duration} seconds reached")
                break
            
            # Log progress periodically
            if frame_count % 300 == 0:  # Every 10 seconds at 30 FPS
                elapsed_time = time.time() - start_time
                logging.info(f"Processed {frame_count} frames in {elapsed_time:.1f} seconds")
                logging.info(f"Current concentration: {metrics.overall_score:.2f} ({metrics.attention_level})")
    
    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
    
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Generate session summary
        summary = analyzer.get_session_summary()
        logging.info("=== Session Summary ===")
        logging.info(f"Duration: {summary.get('session_duration', 0):.1f} seconds")
        logging.info(f"Frames processed: {summary.get('frames_processed', 0)}")
        logging.info(f"Average FPS: {summary.get('avg_fps', 0):.1f}")
        
        concentration_stats = summary.get('concentration_stats', {})
        if concentration_stats:
            logging.info(f"Average concentration: {concentration_stats.get('mean', 0):.2f}")
            logging.info(f"Concentration range: {concentration_stats.get('min', 0):.2f} - {concentration_stats.get('max', 0):.2f}")
        
        # Auto-save session data
        if output_path:
            analyzer.export_session_data(output_path)
        else:
            timestamp = int(time.time())
            auto_save_path = f"session_data_{timestamp}.json"
            analyzer.export_session_data(auto_save_path)
            logging.info(f"Session data auto-saved to {auto_save_path}")


def run_batch_analysis(config: dict, input_path: str, output_path: str):
    """
    Run batch analysis on video file.
    
    Args:
        config: System configuration
        input_path: Path to input video file
        output_path: Path to save results
    """
    logging.info(f"Starting batch analysis on {input_path}")
    
    # Run analysis on video file
    run_real_time_analysis(config, video_source=input_path, output_path=output_path)
    
    logging.info(f"Batch analysis completed, results saved to {output_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Multi-Modal Concentration Analysis System')
    
    parser.add_argument('--config', '-c', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', '-m', choices=['realtime', 'batch'], default='realtime',
                       help='Analysis mode')
    parser.add_argument('--input', '-i', type=str, default=0,
                       help='Input source (webcam index or video file path)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for results')
    parser.add_argument('--duration', '-d', type=int, default=None,
                       help='Maximum duration in seconds')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Convert input to appropriate type
    try:
        video_source = int(args.input)  # Try to convert to webcam index
    except ValueError:
        video_source = args.input  # Use as file path
    
    # Run analysis
    if args.mode == 'realtime':
        run_real_time_analysis(config, video_source, args.duration, args.output)
    elif args.mode == 'batch':
        if isinstance(video_source, int):
            logging.error("Batch mode requires a video file path, not webcam index")
            return
        run_batch_analysis(config, video_source, args.output or 'batch_results.json')


if __name__ == '__main__':
    main()


