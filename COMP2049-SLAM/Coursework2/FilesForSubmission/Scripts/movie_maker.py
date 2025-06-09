#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from glob import glob
from pathlib import Path


def create_video(image_folder, output_path, fps=30, pattern='*.png', sort_numerically=False):
    """
    Create a video from a folder of images.
    
    Args:
        image_folder: Path to folder containing images
        output_path: Path where the video will be saved
        fps: Frames per second for the output video
        pattern: Glob pattern to match image files (e.g., '*.jpg', '*.png')
        sort_numerically: If True, tries to sort files numerically instead of lexicographically
    """
    # Get all matching images in the folder
    image_paths = glob(os.path.join(image_folder, pattern))
    
    if not image_paths:
        print(f"No images found in {image_folder} matching pattern {pattern}")
        return False
    
    # Sort the images
    if sort_numerically:
        # Try to extract numbers from filenames for natural sorting
        def extract_number(filename):
            numbers = ''.join(c for c in os.path.basename(filename) if c.isdigit())
            return int(numbers) if numbers else 0
        
        image_paths.sort(key=extract_number)
    else:
        image_paths.sort()
    
    print(f"Found {len(image_paths)} images")
    
    # Read the first image to get dimensions
    frame = cv2.imread(image_paths[0])
    if frame is None:
        print(f"Could not read first image: {image_paths[0]}")
        return False
    
    height, width, _ = frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each image to the video
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"Warning: Could not read {img_path}")
    
    # Release the video writer
    video.release()
    print(f"Video saved to {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a video from a sequence of images')
    parser.add_argument('--input', '-i', required=True, help='Path to the folder containing images')
    parser.add_argument('--output', '-o', required=True, help='Output video file path')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--pattern', default='*.png', help='Image file pattern (default: *.png)')
    parser.add_argument('--sort-numeric', action='store_true', help='Sort images numerically instead of alphabetically')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    create_video(args.input, args.output, args.fps, args.pattern, args.sort_numeric)
