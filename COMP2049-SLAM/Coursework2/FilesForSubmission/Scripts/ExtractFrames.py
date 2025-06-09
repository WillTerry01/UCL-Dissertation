#!/usr/bin/env python3

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import re

def extract_frames(video_path, output_folder, fps=None, focal_length=None, cx=None, cy=None, k=None, rename_sequentially=True, resize_to_720p=False):
    """
    Extract frames from a video, convert to grayscale, and optionally undistort them using 
    the Simple Radial camera model parameters. Also extracts timestamps.
    
    Args:
        video_path (str): Path to input video
        output_folder (str): Path to save extracted frames
        fps (float): If specified, extract frames at this rate. Otherwise extract all frames
        focal_length (float): Focal length parameter f (for undistortion)
        cx (float): Principal point x coordinate (for undistortion)
        cy (float): Principal point y coordinate (for undistortion)
        k1 (float): Radial distortion parameter (for undistortion)
        k2 (float): Radial distortion parameter (for undistortion)
        p1 (float): Tangential distortion parameter (for undistortion)
        p2 (float): Tangential distortion parameter (for undistortion)
        rename_sequentially (bool): If True, rename images as 000000.png, 000001.png, etc.
        resize_to_720p (bool): If True, resize output frames to 1280x720 resolution
    
    Returns:
        tuple: (frame_count, new camera parameters if undistortion was performed)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create image_0 directory for frame storage
    image_dir = os.path.join(output_folder, "image_0")
    os.makedirs(image_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {video_fps}")
    print(f"  Total frames: {total_frames}")
    
    # If fps is specified, calculate frame interval
    if fps and fps < video_fps:
        frame_interval = int(video_fps / fps)
        expected_frames = total_frames // frame_interval
        print(f"Extracting at {fps} FPS (every {frame_interval} frames)")
        print(f"Expected output frames: ~{expected_frames}")
    else:
        frame_interval = 1
        print(f"Extracting all frames at original {video_fps} FPS")
    
    # Setup for undistortion if parameters are provided
    undistort = all([focal_length is not None, cx is not None, cy is not None, k is not None])
    new_camera_params = None
    
    if undistort:
        # Create camera matrix for the Simple Radial model
        camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        
        # Create distortion coefficients array (k1, k2, p1, p2, k3)
        # For Simple Radial model, only k1 (the first coefficient) is used
        dist_coeffs = np.array([k, 0, 0, 0, 0])
        
        # Calculate optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (width, height), 0, (width, height)
        )
        
        # Extract the new camera parameters (pinhole model: fx, fy, cx, cy)
        fx = new_camera_matrix[0, 0]
        fy = new_camera_matrix[1, 1]
        new_cx = new_camera_matrix[0, 2]
        new_cy = new_camera_matrix[1, 2]
        
        new_camera_params = {
            'fx': fx,
            'fy': fy,
            'cx': new_cx,
            'cy': new_cy,
            'roi': roi
        }
        
        print(f"Original Simple Radial parameters: f={focal_length}, cx={cx}, cy={cy}, k={k}")
        print(f"New Pinhole parameters: fx={fx}, fy={fy}, cx={new_cx}, cy={new_cy}")
    
    # Set resize parameters if needed
    if resize_to_720p:
        target_width = 1280
        target_height = 720
        print(f"Output frames will be resized to 720p (1280x720)")
        
        # Adjust camera parameters if undistortion was performed
        if undistort and new_camera_params:
            # Calculate scaling factors
            scale_x = target_width / width if width > 0 else 1
            scale_y = target_height / height if height > 0 else 1
            
            # Scale the camera parameters
            new_camera_params['fx'] *= scale_x
            new_camera_params['fy'] *= scale_y
            new_camera_params['cx'] *= scale_x
            new_camera_params['cy'] *= scale_y
            
            # Update ROI if it exists
            if 'roi' in new_camera_params:
                x, y, w, h = new_camera_params['roi']
                new_camera_params['roi'] = (
                    int(x * scale_x), 
                    int(y * scale_y), 
                    int(w * scale_x), 
                    int(h * scale_y)
                )
            
            print(f"Scaled camera parameters: fx={new_camera_params['fx']}, fy={new_camera_params['fy']}, "
                  f"cx={new_camera_params['cx']}, cy={new_camera_params['cy']}")
    
    # Process video
    frame_count = 0
    saved_count = 0
    timestamps = []
    
    while True:
        # Read the next frame
        success, frame = video.read()
        if not success:
            break
        
        # Check if we should process this frame
        if frame_count % frame_interval == 0:
            # Get timestamp in milliseconds and convert to seconds
            timestamp_sec = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamps.append(timestamp_sec)
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Undistort if parameters were provided
            if undistort:
                # Undistort the image
                undistorted_frame = cv2.undistort(gray_frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
                
                # Crop the image using the ROI
                x, y, w, h = roi
                if w > 0 and h > 0:  # Ensure valid ROI
                    undistorted_frame = undistorted_frame[y:y+h, x:x+w]
                
                processed_frame = undistorted_frame
            else:
                processed_frame = gray_frame
            
            # Resize to 720p if requested
            if resize_to_720p:
                processed_frame = cv2.resize(processed_frame, (1280, 720), interpolation=cv2.INTER_AREA)
            
            # Save the frame
            if rename_sequentially:
                output_path = os.path.join(image_dir, f"{saved_count:06d}.png")
            else:
                output_path = os.path.join(image_dir, f"frame_{frame_count:06d}.png")
            
            cv2.imwrite(output_path, processed_frame)
            saved_count += 1
            
            # Print progress
            if saved_count % 100 == 0:
                print(f"Processed {saved_count} frames...")
        
        frame_count += 1
    
    # Clean up
    video.release()
    
    print(f"Extraction completed. {saved_count} frames saved to {image_dir}")
    
    # Save timestamps to file
    timestamps_file = os.path.join(output_folder, "times.txt")
    with open(timestamps_file, 'w') as f:
        for ts in timestamps:
            f.write(f"{ts:e}\n")
    print(f"Saved {len(timestamps)} timestamps to {timestamps_file}")
    
    # Save the camera parameters to a file if undistortion was performed
    if undistort:
        params_file = os.path.join(output_folder, "camera_parameters.txt")
        with open(params_file, 'w') as f:
            f.write("# Original Simple Radial camera parameters\n")
            f.write(f"f: {focal_length}\n")
            f.write(f"cx: {cx}\n")
            f.write(f"cy: {cy}\n")
            f.write(f"k: {k}\n\n")
            f.write("# New Pinhole camera parameters\n")
            
            if resize_to_720p:
                f.write("# Note: Parameters have been scaled for 720p (1280x720) output\n")
                
            f.write(f"fx: {new_camera_params['fx']}\n")
            f.write(f"fy: {new_camera_params['fy']}\n")
            f.write(f"cx: {new_camera_params['cx']}\n")
            f.write(f"cy: {new_camera_params['cy']}\n")
        
        print(f"Camera parameters saved to {params_file}")
    
    return saved_count, new_camera_params

def read_colmap_camera(cameras_file):
    """
    Read camera parameters from COLMAP cameras.txt file.
    
    Args:
        cameras_file (str): Path to cameras.txt file from COLMAP
    
    Returns:
        dict: Camera parameters for Simple Radial model
    """
    with open(cameras_file, 'r') as f:
        lines = f.readlines()
        
    # Skip header lines
    for line in lines:
        if not line.startswith('#'):
            params = line.strip().split()
            break
    
    camera_id = int(params[0])
    model = params[1]
    width = int(params[2])
    height = int(params[3])
    
    if model == "SIMPLE_RADIAL":
        # SIMPLE_RADIAL format: f, cx, cy, k
        focal_length = float(params[4])
        cx = float(params[5])
        cy = float(params[6])
        k = float(params[7])
        
        return {
            'model': model,
            'width': width,
            'height': height,
            'focal_length': focal_length,
            'cx': cx,
            'cy': cy,
            'k': k
        }
    else:
        raise ValueError(f"Camera model {model} is not SIMPLE_RADIAL, cannot proceed with undistortion")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from a video, convert to grayscale, and optionally undistort')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('output_folder', help='Path to output folder for extracted frames')
    parser.add_argument('--fps', type=float, help='Extract frames at this FPS rate (default: extract all frames)')
    parser.add_argument('--cameras-file', help='Path to COLMAP cameras.txt file with SIMPLE_RADIAL parameters')
    parser.add_argument('--focal-length', type=float, help='Focal length parameter f')
    parser.add_argument('--cx', type=float, help='Principal point x coordinate')
    parser.add_argument('--cy', type=float, help='Principal point y coordinate')
    parser.add_argument('--k', type=float, help='Radial distortion parameter')
    parser.add_argument('--no-sequential', action='store_false', dest='rename_sequentially', 
                        help='Do not rename frames sequentially (use frame_{frame_count:06d}.png instead)')
    parser.add_argument('--resize', action='store_true', dest='resize_to_720p', 
                        help='Resize output frames to 1280x720 resolution')
    args = parser.parse_args()
    
    try:
        # Check if camera parameters are provided directly or via cameras file
        focal_length, cx, cy, k = None, None, None, None
        
        if args.cameras_file:
            camera_params = read_colmap_camera(args.cameras_file)
            focal_length = camera_params['focal_length']
            cx = camera_params['cx']
            cy = camera_params['cy']
            k = camera_params['k']
        elif all([args.focal_length, args.cx, args.cy, args.k]):
            focal_length = args.focal_length
            cx = args.cx
            cy = args.cy
            k = args.k
        elif any([args.focal_length, args.cx, args.cy, args.k]):
            # If some but not all parameters are provided
            raise ValueError("For undistortion, you must provide all camera parameters (--focal-length, --cx, --cy, --k) or use --cameras-file")
        
        # Extract frames
        extract_frames(
            args.video_path, 
            args.output_folder,
            args.fps,
            focal_length,
            cx,
            cy,
            k,
            args.rename_sequentially,
            args.resize_to_720p
        )
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
