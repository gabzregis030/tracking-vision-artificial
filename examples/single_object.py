"""
Single object tracking example.

This example demonstrates basic single object tracking using manual ROI selection.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers import AppController


def main():
    """Run single object tracking example."""
    print("=== Single Object Tracking Example ===\n")
    
    # Ask for video source
    print("Select video source:")
    print("  1. Use webcam")
    print("  2. Use video file")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        video_source = 0
        print("\nUsing default webcam (index 0)")
    else:
        video_path = input("Enter path to video file: ").strip()
        if not Path(video_path).exists():
            print(f"Error: Video file not found: {video_path}")
            return
        video_source = video_path
    
    # Ask for tracker type
    print("\nAvailable trackers:")
    print("  1. CSRT (High accuracy, slower)")
    print("  2. KCF (Balanced)")
    print("  3. MOSSE (Very fast, lower accuracy)")
    print("  4. MEDIANFLOW (Good for predictable motion)")
    
    tracker_choice = input("\nSelect tracker (1-4, default 1): ").strip() or '1'
    tracker_map = {'1': 'CSRT', '2': 'KCF', '3': 'MOSSE', '4': 'MEDIANFLOW'}
    tracker_type = tracker_map.get(tracker_choice, 'CSRT')
    
    # Ask for Kalman filtering
    use_kalman = input("\nUse Kalman filtering for smoother tracking? (y/n, default n): ").strip().lower() == 'y'
    
    # Create controller
    controller = AppController(
        video_source=video_source,
        tracker_type=tracker_type,
        use_kalman=use_kalman
    )
    
    # Initialize
    if not controller.initialize():
        print("Failed to initialize controller")
        return
    
    # Select object to track
    if not controller.select_objects(num_objects=1):
        print("Failed to select object")
        return
    
    # Run tracking
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nTracking interrupted by user")
    finally:
        controller.cleanup()


if __name__ == '__main__':
    main()
