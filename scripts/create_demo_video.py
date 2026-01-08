"""
Generate a simple demo video with two moving objects for testing the tracker.
"""
import cv2
import numpy as np
from pathlib import Path

def create_demo_video():
    """Create a demo video with two moving colored balls."""
    
    # Video parameters
    width, height = 640, 480
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Output path
    output_path = Path(__file__).parent.parent / 'videos' / 'demo_two_objects.mp4'
    output_path.parent.mkdir(exist_ok=True)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Object 1: Red ball (moves horizontally)
    obj1_color = (0, 0, 255)  # Red in BGR
    obj1_radius = 30
    obj1_start_x = 50
    obj1_y = height // 3
    
    # Object 2: Blue ball (moves in circle)
    obj2_color = (255, 0, 0)  # Blue in BGR
    obj2_radius = 25
    center_x = width // 2
    center_y = 2 * height // 3
    circle_radius = 100
    
    print(f"Creating demo video: {output_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Duration: {duration} seconds ({total_frames} frames)")
    
    for frame_num in range(total_frames):
        # Create blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Draw grid for reference
        for i in range(0, width, 50):
            cv2.line(frame, (i, 0), (i, height), (220, 220, 220), 1)
        for i in range(0, height, 50):
            cv2.line(frame, (0, i), (width, i), (220, 220, 220), 1)
        
        # Object 1: Horizontal movement (back and forth)
        progress = frame_num / total_frames
        obj1_x = int(obj1_start_x + (width - 2 * obj1_start_x) * abs(np.sin(progress * np.pi * 2)))
        cv2.circle(frame, (obj1_x, obj1_y), obj1_radius, obj1_color, -1)
        cv2.circle(frame, (obj1_x, obj1_y), obj1_radius, (0, 0, 0), 2)  # Border
        
        # Object 2: Circular movement
        angle = progress * 2 * np.pi * 2  # 2 full circles
        obj2_x = int(center_x + circle_radius * np.cos(angle))
        obj2_y = int(center_y + circle_radius * np.sin(angle))
        cv2.circle(frame, (obj2_x, obj2_y), obj2_radius, obj2_color, -1)
        cv2.circle(frame, (obj2_x, obj2_y), obj2_radius, (0, 0, 0), 2)  # Border
        
        # Add text info
        cv2.putText(frame, f"Frame: {frame_num + 1}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, "Red ball: Horizontal motion", 
                   (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        cv2.putText(frame, "Blue ball: Circular motion", 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)
        
        out.write(frame)
        
        # Progress indicator
        if (frame_num + 1) % 30 == 0:
            print(f"Progress: {frame_num + 1}/{total_frames} frames ({(frame_num + 1) / total_frames * 100:.1f}%)")
    
    out.release()
    print(f"\n✅ Demo video created successfully: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\nYou can now test tracking with:")
    print(f"python src/main.py --video {output_path} --multi --num-objects 2 --tracker KCF")

if __name__ == '__main__':
    create_demo_video()
