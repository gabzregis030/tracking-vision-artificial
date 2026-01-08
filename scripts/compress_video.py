import cv2
import argparse
import os

def compress_video(input_path, output_path, scale_percent=50, target_fps=30):
    """
    Compress video by resizing and adjusting FPS.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get original video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate new dimensions
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    
    print(f"Original: {width}x{height} @ {fps}fps")
    print(f"Target: {new_width}x{new_height} @ {target_fps}fps")

    # Define codec and create VideoWriter
    # mp4v is a good general option, h264 would be better if available but setup varies
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (new_width, new_height))

    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Write frame
        out.write(resized_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...", end='\r')

    cap.release()
    out.release()
    print(f"\nVideo compressed and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress video for testing")
    parser.add_argument("input_file", help="Path to input video file")
    parser.add_argument("--output", help="Path to output video file (optional)")
    parser.add_argument("--scale", type=int, default=50, help="Scale percentage (default: 50)")
    parser.add_argument("--fps", type=int, default=20, help="Target FPS (default: 20)")
    
    args = parser.parse_args()
    
    input_video = args.input_file
    if args.output:
        output_video = args.output
    else:
        # Auto-generate output name
        base, ext = os.path.splitext(input_video)
        output_video = f"{base}_compressed{ext}"

    compress_video(input_video, output_video, args.scale, args.fps)
