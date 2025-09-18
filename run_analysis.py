"""
Simple launcher script for Multi-Modal Concentration Analysis System

This script provides a simple way to run the concentration analysis system
without import issues.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

# Now we can import our modules
try:
    import cv2
    import numpy as np
    import yaml
    import logging
    import time
    from datetime import datetime
    
    # Import our modules
    from integration.concentration_analyzer import ConcentrationAnalyzer
    
    print("✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease make sure you have installed all dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('concentration_analysis.log')
        ]
    )


def get_default_config():
    """Get default configuration."""
    return {
        'gaze': {
            'device': 'cpu',
            'method': 'gazenet'
        },
        'blink': {
            'method': 'ear',
            'ear_threshold': 0.25,
            'consecutive_frames': 3
        },
        'drowsiness': {
            'perclos_threshold': 0.8,
            'analysis_window': 60.0,
            'drowsy_blink_rate_low': 5.0,
            'drowsy_blink_rate_high': 25.0
        },
        'head_pose': {
            'method': 'pnp',
            'smoothing_window': 5
        },
        'fusion': {
            'method': 'weighted_average',
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
            'save_video': False
        }
    }


def run_concentration_analysis(duration=None):
    """Run the concentration analysis system."""
    
    print("🎯 Starting Multi-Modal Concentration Analysis System")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Get configuration
    config = get_default_config()
    
    try:
        # Initialize concentration analyzer
        logger.info("Initializing concentration analyzer...")
        analyzer = ConcentrationAnalyzer(config)
        
        # Setup video capture
        logger.info("Setting up webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            print("Please check:")
            print("1. Webcam is connected and working")
            print("2. No other applications are using the webcam")
            print("3. Camera permissions are granted")
            return False
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"✅ Webcam initialized: {frame_width}x{frame_height} @ {fps} FPS")
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save session data")
        print("- Press 'r' to reset analysis")
        print("\nStarting analysis in 3 seconds...")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        print("🚀 Analysis started!")
        
        # Analysis loop
        start_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from webcam")
                break
            
            try:
                # Analyze frame
                metrics = analyzer.analyze_frame(frame)
                
                # Visualize results
                result_frame = analyzer.visualize_analysis(frame, metrics)
                
                # Display frame
                cv2.imshow('Concentration Analysis', result_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('s'):
                    # Save session data
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"session_data_{timestamp}.json"
                    if analyzer.export_session_data(save_path):
                        print(f"✅ Session data saved to {save_path}")
                    else:
                        print("❌ Failed to save session data")
                elif key == ord('r'):
                    # Reset analysis
                    analyzer = ConcentrationAnalyzer(config)
                    logger.info("Analysis reset")
                    print("🔄 Analysis reset")
                
                frame_count += 1
                
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    logger.info(f"Duration limit of {duration} seconds reached")
                    break
                
                # Log progress periodically
                if frame_count % 300 == 0:  # Every 10 seconds at 30 FPS
                    elapsed_time = time.time() - start_time
                    current_score = metrics.overall_score
                    current_level = metrics.attention_level
                    logger.info(f"Progress: {frame_count} frames, {elapsed_time:.1f}s")
                    logger.info(f"Current: {current_score:.2f} ({current_level})")
                    print(f"📊 Current concentration: {current_score:.2f} ({current_level})")
            
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Generate session summary
        summary = analyzer.get_session_summary()
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Duration: {summary.get('session_duration', 0):.1f} seconds")
        print(f"Frames processed: {summary.get('frames_processed', 0)}")
        print(f"Average FPS: {summary.get('avg_fps', 0):.1f}")
        
        concentration_stats = summary.get('concentration_stats', {})
        if concentration_stats:
            print(f"Average concentration: {concentration_stats.get('mean', 0):.2f}")
            print(f"Concentration range: {concentration_stats.get('min', 0):.2f} - {concentration_stats.get('max', 0):.2f}")
        
        # Auto-save session data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_save_path = f"session_data_{timestamp}.json"
        analyzer.export_session_data(auto_save_path)
        print(f"📁 Session data saved to {auto_save_path}")
        
        return True
    
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return True
    
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        print(f"❌ Error: {e}")
        return False


def main():
    """Main function."""
    print("🎯 Multi-Modal Concentration Analysis System")
    print("=" * 60)
    
    try:
        # Run analysis
        success = run_concentration_analysis()
        
        if success:
            print("\n✅ Analysis completed successfully!")
        else:
            print("\n❌ Analysis encountered issues.")
            return 1
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
