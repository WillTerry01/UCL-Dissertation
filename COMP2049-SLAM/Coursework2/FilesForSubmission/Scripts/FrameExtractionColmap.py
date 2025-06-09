#!/usr/bin/env python3

import cv2
import os
import argparse

def extract_frames(video_path, output_folder, resize_width=None, resize_height=None, target_fps=None):
    """
    Extract frames from a video and resize them to the specified dimensions.

    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to the output folder where frames will be saved
        resize_width (int, optional): Width to resize frames to
        resize_height (int, optional): Height to resize frames to 
        target_fps (float, optional): Target FPS for frame extraction
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    original_fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video FPS: {original_fps}")
    print(f"Total frames: {frame_count}")
    print(f"Output folder: {output_folder}")
    
    if resize_width and resize_height:
        print(f"Resizing frames to: {resize_width}x{resize_height}")
    
    # Calculate frame extraction rate
    if target_fps and target_fps < original_fps:
        # Calculate how many original frames to skip
        frame_interval = original_fps / target_fps
        print(f"Target FPS: {target_fps}")
        print(f"Will extract approximately 1 frame every {frame_interval:.2f} frames")
    else:
        frame_interval = 1
        if target_fps:
            print(f"Warning: Target FPS ({target_fps}) is greater than or equal to video FPS ({original_fps})")
            print("Extracting all frames")
        else:
            print("No target FPS specified, extracting all frames")
    
    # Read and process frames
    frame_number = 0
    saved_count = 0
    next_frame_to_save = 0
    
    while True:
        ret, frame = video.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        # Determine if we should save this frame
        if frame_number >= next_frame_to_save:
            # Resize frame if dimensions are provided
            if resize_width and resize_height:
                frame = cv2.resize(frame, (resize_width, resize_height))
            
            # Save the frame
            frame_path = os.path.join(output_folder, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
            # Calculate next frame to save
            next_frame_to_save += frame_interval
            
            # Print progress every 100 frames
            if saved_count % 100 == 0:
                print(f"Saved {saved_count} frames...")
        
        frame_number += 1
    
    # Release video capture
    video.release()
    
    print(f"Extraction complete! Saved {saved_count} frames to {output_folder}")
    if target_fps:
        print(f"Effective output FPS: {saved_count / (frame_count / original_fps):.2f}")

def main():
    parser = argparse.ArgumentParser(description="Extract and resize frames from a video.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("output_folder", help="Path to the output folder for extracted frames")
    parser.add_argument("--width", type=int, help="Width to resize frames to")
    parser.add_argument("--height", type=int, help="Height to resize frames to")
    parser.add_argument("--fps", type=float, help="Target FPS for frame extraction")
    
    args = parser.parse_args()
    
    # If only one dimension is provided, calculate the other to maintain aspect ratio
    if (args.width is None) != (args.height is None):
        print("Error: Please provide both width and height for resizing.")
        return
    
    extract_frames(
        args.video_path,
        args.output_folder,
        args.width,
        args.height,
        args.fps
    )

if __name__ == "__main__":
    main()
