"""
Webcam tracking example with real-time performance metrics.

This example demonstrates real-time tracking using webcam.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from controllers import AppController


def main():
    """Run webcam tracking example."""
    print("=== Webcam Real-Time Tracking Example ===\n")
    
    # Camera selection
    camera_index = int(input("Enter camera index (0 for default, default 0): ").strip() or '0')
    
    # For webcam, we recommend MOSSE for speed
    print("\nRecommended tracker for webcam: MOSSE (fastest)")
    print("Alternative: KCF (more accurate but slower)")
    
    use_mosse = input("Use MOSSE tracker? (y/n, default y): ").strip().lower() != 'n'
    tracker_type = 'MOSSE' if use_mosse else 'KCF'
    
    # Number of objects
    num_objects = int(input("\nHow many objects to track? (default 1): ").strip() or '1')
    
    # Kalman filtering
    use_kalman = input("Use Kalman filtering? (y/n, default y): ").strip().lower() != 'n'
    
    print(f"\nStarting webcam tracking...")
    print(f"  Camera: {camera_index}")
    print(f"  Tracker: {tracker_type}")
    print(f"  Objects: {num_objects}")
    print(f"  Kalman: {'Yes' if use_kalman else 'No'}")
    
    # Create controller
    controller = AppController(
        video_source=camera_index,
        tracker_type=tracker_type,
        use_kalman=use_kalman
    )
    
    # Initialize
    if not controller.initialize():
        print("Failed to initialize controller")
        print("\nTroubleshooting:")
        print("  - Check if camera is connected")
        print("  - Try a different camera index")
        print("  - Close other applications using the camera")
        return
    
    # Select objects to track
    print("\nPosition object(s) in view of camera, then select them...")
    if not controller.select_objects(num_objects=num_objects):
        print("Failed to select objects")
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
