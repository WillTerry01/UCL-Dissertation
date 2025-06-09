#!/usr/bin/env python3

import os
import numpy as np
import cv2
import argparse
from pathlib import Path

def extract_timestamps_from_video(video_path, frame_interval=8, output_file=None):
    """
    Extract timestamps from a video file at regular intervals.
    
    Args:
        video_path (str): Path to the video file
        frame_interval (int): Extract timestamp for every nth frame
        output_file (str): Path to save the timestamps (if None, prints to console)
    
    Returns:
        list: List of timestamps in seconds
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    frame_count = 0
    saved_count = 0
    timestamps = []
    
    print(f"Extracting timestamps from {video_path}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Get timestamp in milliseconds and convert to seconds
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamps.append(timestamp_sec)
            saved_count += 1
            print(f"Extracted {saved_count} timestamps")
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} timestamps")
    
    # Save timestamps to file if output_file is provided
    if output_file:
        with open(output_file, 'w') as f:
            for ts in timestamps:
                f.write(f"{ts:e}\n")
        print(f"Saved {len(timestamps)} timestamps to {output_file}")
    
    return timestamps

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract timestamps from a video file')
    parser.add_argument('video_path', help='Path to the input video file (.mov)')
    parser.add_argument('--frame-interval', type=int, default=8, 
                        help='Extract timestamp for every nth frame (default: 8)')
    parser.add_argument('--output-file', default='times.txt',
                        help='Path to save the timestamps (default: times.txt)')
    args = parser.parse_args()
    
    # Extract timestamps
    try:
        extract_timestamps_from_video(args.video_path, args.frame_interval, args.output_file)
    except Exception as e:
        print(f"Error extracting timestamps: {e}")
        return

if __name__ == "__main__":
    main()
