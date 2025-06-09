#!/usr/bin/env python3

import os
import subprocess
import numpy as np
from pathlib import Path
import cv2
import argparse

def extract_frames_from_video(video_path, output_dir, frame_interval=30):
    """
    Extract frames from a video file at regular intervals.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
        frame_interval (int): Extract every nth frame
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    frame_count = 0
    saved_count = 0
    
    print(f"Extracting frames from {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Save frame as image
            frame_path = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")
    return saved_count

def resize_images_to_720p(image_dir):
    """
    Resize all images in the directory to 720p while maintaining aspect ratio.
    
    Args:
        image_dir (str): Directory containing images to resize
    """
    target_height = 720
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Resizing {len(image_files)} images to 720p...")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        h, w = img.shape[:2]
        # Calculate new width to maintain aspect ratio
        new_width = int(w * (target_height / h))
        
        # Resize the image
        resized_img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Save the resized image (overwrite original)
        cv2.imwrite(img_path, resized_img)
    
    print(f"Resized {len(image_files)} images to 720p")

def run_colmap_calibration(image_dir, output_dir):
    """
    Run COLMAP calibration on a set of images to find camera intrinsics.
    
    Args:
        image_dir (str): Directory containing the calibration images
        output_dir (str): Directory to store COLMAP outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run feature extractor
    feature_extractor_cmd = [
        'colmap', 'feature_extractor',
        '--database_path', os.path.join(output_dir, 'database.db'),
        '--image_path', image_dir,
        '--ImageReader.single_camera', '1'
    ]
    subprocess.run(feature_extractor_cmd, check=True)
    
    # Run exhaustive matcher
    matcher_cmd = [
        'colmap', 'exhaustive_matcher',
        '--database_path', os.path.join(output_dir, 'database.db'),
        '--SiftMatching.guided_matching', '1',
        '--SiftMatching.max_ratio', '0.8',
        '--SiftMatching.max_distance', '0.7',
        '--SiftMatching.cross_check', '1'
    ]
    subprocess.run(matcher_cmd, check=True)
    
    # Run mapper
    mapper_cmd = [
        'colmap', 'mapper',
        '--database_path', os.path.join(output_dir, 'database.db'),
        '--image_path', image_dir,
        '--output_path', output_dir,
        '--Mapper.min_num_matches', '15',
        '--Mapper.init_min_num_inliers', '15',
        '--Mapper.multiple_models', '0',
        '--Mapper.extract_colors', '1'
    ]
    subprocess.run(mapper_cmd, check=True)

def extract_camera_intrinsics(output_dir):
    """
    Extract camera intrinsics from COLMAP's output.
    
    Args:
        output_dir (str): Directory containing COLMAP outputs
    """
    # Read the cameras.bin file
    cameras_file = os.path.join(output_dir, 'sparse', '0', 'cameras.bin')
    if not os.path.exists(cameras_file):
        print("Error: cameras.bin not found. Calibration may have failed.")
        return None
    
    # Use COLMAP's model_converter to convert binary to text
    converter_cmd = [
        'colmap', 'model_converter',
        '--input_path', os.path.join(output_dir, 'sparse', '0'),
        '--output_path', os.path.join(output_dir, 'sparse', '0', 'cameras.txt'),
        '--output_type', 'TXT'
    ]
    subprocess.run(converter_cmd, check=True)
    
    # Read the camera parameters
    with open(os.path.join(output_dir, 'sparse', '0', 'cameras.txt'), 'r') as f:
        lines = f.readlines()
        # Skip header line
        camera_params = lines[1].strip().split()
        
        # Extract intrinsics
        camera_id, model, width, height, *params = camera_params
        params = [float(p) for p in params]
        
        if model == 'SIMPLE_PINHOLE':
            fx, cx, cy = params
            print("\nCamera Intrinsics (SIMPLE_PINHOLE model):")
            print(f"Focal length (fx): {fx:.2f}")
            print(f"Principal point (cx, cy): ({cx:.2f}, {cy:.2f})")
            print(f"Image dimensions: {width}x{height}")
        else:
            print(f"\nCamera model: {model}")
            print("Parameters:", params)
            print(f"Image dimensions: {width}x{height}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract camera intrinsics from a video file using COLMAP')
    parser.add_argument('video_path', help='Path to the input video file (.mov)')
    parser.add_argument('--frame-interval', type=int, default=30, help='Extract every nth frame (default: 30)')
    parser.add_argument('--resize', action='store_true', help='Resize extracted frames to 720p')
    args = parser.parse_args()
    
    # Get the current directory
    current_dir = os.getcwd()
    
    # Define directories
    frames_dir = os.path.join(current_dir, 'extracted_frames')
    output_dir = os.path.join(current_dir, 'colmap_output')
    
    # Extract frames from video
    try:
        num_frames = extract_frames_from_video(args.video_path, frames_dir, args.frame_interval)
        if num_frames < 20:
            print("Warning: Less than 20 frames were extracted. This might not be enough for good calibration.")
            print("Consider reducing the frame interval to get more frames.")
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return
    
    # Resize images if requested
    if args.resize:
        resize_images_to_720p(frames_dir)
    
    print("\nStarting COLMAP calibration process...")
    run_colmap_calibration(frames_dir, output_dir)
    print("\nCalibration completed. Extracting camera intrinsics...")
    extract_camera_intrinsics(output_dir)

if __name__ == "__main__":
    main()
