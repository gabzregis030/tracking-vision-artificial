"""
Vision Tracking Project - Main Entry Point

Track objects in video using various tracking algorithms.
"""

import argparse
import sys
from pathlib import Path

from controllers import AppController


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Object tracking with multiple algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Track object in video with CSRT tracker
  python main.py --video path/to/video.mp4 --tracker CSRT
  
  # Track using webcam with MOSSE tracker
  python main.py --camera 0 --tracker MOSSE
  
  # Track multiple objects
  python main.py --video path/to/video.mp4 --multi --num-objects 3
  
  # Use Kalman filtering for smoother tracking
  python main.py --video path/to/video.mp4 --kalman
  
  # Auto-detect and track objects
  python main.py --video path/to/video.mp4 --detect --detector yolo
        """
    )
    
    # Video source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--video', type=str, help='Path to video file')
    source_group.add_argument('--camera', type=int, help='Camera index (0 for default webcam)')
    
    # Tracker options
    parser.add_argument('--tracker', type=str, default='CSRT',
                       choices=['CSRT', 'KCF', 'MOSSE', 'MEDIANFLOW', 'MIL', 'BOOSTING'],
                       help='Tracker algorithm to use (default: CSRT)')
    
    # Multi-object tracking
    parser.add_argument('--multi', action='store_true',
                       help='Enable multi-object tracking')
    parser.add_argument('--num-objects', type=int, default=1,
                       help='Number of objects to track (default: 1)')
    
    # Advanced options
    parser.add_argument('--kalman', action='store_true',
                       help='Use Kalman filter for smoothing')
    
    # Detection options
    parser.add_argument('--detect', action='store_true',
                       help='Auto-detect objects (requires detector)')
    parser.add_argument('--detector', type=str, default='yolo',
                       choices=['yolo', 'hog'],
                       help='Detector to use for auto-detection (default: yolo)')
    parser.add_argument('--detect-classes', type=str, nargs='+',
                       help='Classes to detect (e.g., person car)')
    
    return parser.parse_args()


def main():
    """Main application entry point."""
    args = parse_arguments()
    
    # Determine video source
    if args.video:
        if not Path(args.video).exists():
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        video_source = args.video
    else:
        video_source = args.camera
    
    # Create controller
    print("=== Vision Tracking System ===")
    print(f"Tracker: {args.tracker}")
    print(f"Kalman Filtering: {'Enabled' if args.kalman else 'Disabled'}")
    
    controller = AppController(
        video_source=video_source,
        tracker_type=args.tracker,
        use_kalman=args.kalman,
        use_detector=args.detect,
        detector_method=args.detector if args.detect else 'yolo'
    )
    
    # Initialize
    if not controller.initialize():
        print("Failed to initialize tracking system")
        sys.exit(1)
    
    # Initialize tracking
    if args.detect:
        # Auto-detect objects
        print("\nAuto-detecting objects...")
        if not controller.detect_and_track(args.detect_classes):
            print("Failed to detect objects. Falling back to manual selection...")
            if not controller.select_objects(args.num_objects):
                print("Failed to select objects")
                sys.exit(1)
    else:
        # Manual object selection
        if not controller.select_objects(args.num_objects):
            print("Failed to select objects")
            sys.exit(1)
    
    # Run tracking
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n\nTracking interrupted by user")
    except Exception as e:
        print(f"\n\nError during tracking: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.cleanup()
    
    print("\nThank you for using Vision Tracking System!")


if __name__ == '__main__':
    main()
