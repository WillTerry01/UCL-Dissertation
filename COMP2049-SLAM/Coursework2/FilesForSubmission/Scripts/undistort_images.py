#!/usr/bin/env python3

import os
import cv2
import numpy as np
import argparse
import glob
from pathlib import Path
import re

def undistort_images(input_folder, output_folder, focal_length, cx, cy, k, rename_sequentially=False):
    """
    Undistort images using the Simple Radial camera model parameters and convert to Pinhole model.
    
    Args:
        input_folder (str): Path to the folder containing distorted images
        output_folder (str): Path to save undistorted images
        focal_length (float): Focal length parameter f
        cx (float): Principal point x coordinate
        cy (float): Principal point y coordinate
        k (float): Radial distortion parameter
        rename_sequentially (bool): If True, rename images as 000000.png, 000001.png, etc.
    
    Returns:
        str: Path to the output folder
    """
    # Get absolute path of input folder
    input_folder = os.path.abspath(input_folder)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get a list of image files in the input folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    # Sort the files naturally
    def natural_sort_key(s):
        numbers = re.findall(r'\d+', os.path.basename(s))
        if numbers:
            return int(numbers[0])
        return os.path.basename(s)
    
    try:
        image_files.sort(key=natural_sort_key)
    except:
        image_files.sort()
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return output_folder
    
    print(f"Found {len(image_files)} images in {input_folder}")
    
    # Get image size from the first image
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        raise ValueError(f"Could not read first image: {image_files[0]}")
    
    height, width = first_img.shape[:2]
    
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
    
    print(f"Original Simple Radial parameters: f={focal_length}, cx={cx}, cy={cy}, k={k}")
    print(f"New Pinhole parameters: fx={fx}, fy={fy}, cx={new_cx}, cy={new_cy}")
    
    # Process each image
    for i, img_path in enumerate(image_files):
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Undistort the image
            undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
            
            # Crop the image using the ROI
            x, y, w, h = roi
            if w > 0 and h > 0:  # Ensure valid ROI
                undistorted_img = undistorted_img[y:y+h, x:x+w]
            
            # Create output path with either sequential name or original name
            if rename_sequentially:
                filename = f"{i:06d}.png"  # Format: 000000.png, 000001.png, etc.
            else:
                filename = os.path.basename(img_path)
                
            output_path = os.path.join(output_folder, filename)
            
            # Save undistorted image
            cv2.imwrite(output_path, undistorted_img)
            
            # Print progress
            if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                print(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Save the camera parameters to a file
    params_file = os.path.join(output_folder, "camera_parameters.txt")
    with open(params_file, 'w') as f:
        f.write("# Original Simple Radial camera parameters\n")
        f.write(f"f: {focal_length}\n")
        f.write(f"cx: {cx}\n")
        f.write(f"cy: {cy}\n")
        f.write(f"k: {k}\n\n")
        f.write("# New Pinhole camera parameters\n")
        f.write(f"fx: {fx}\n")
        f.write(f"fy: {fy}\n")
        f.write(f"cx: {new_cx}\n")
        f.write(f"cy: {new_cy}\n")
    
    print(f"Undistortion completed. Images saved to {output_folder}")
    print(f"Camera parameters saved to {params_file}")
    
    return output_folder

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

def process_kitti_dataset(dataset_path, cameras_file, rename_sequentially=False):
    """
    Process a KITTI-format dataset, undistorting all image sequences.
    
    Args:
        dataset_path (str): Path to the KITTI dataset (containing sequences folder)
        cameras_file (str): Path to COLMAP cameras.txt file with SIMPLE_RADIAL parameters
        rename_sequentially (bool): If True, rename images as 000000.png, 000001.png, etc.
    """
    # Read camera parameters
    camera_params = read_colmap_camera(cameras_file)
    
    sequences_path = os.path.join(dataset_path, "sequences")
    if not os.path.exists(sequences_path):
        raise ValueError(f"Sequences folder not found at {sequences_path}")
    
    # Find all sequence folders
    sequence_folders = [f for f in os.listdir(sequences_path) if os.path.isdir(os.path.join(sequences_path, f))]
    print(f"Found {len(sequence_folders)} sequence folders")
    
    for seq in sequence_folders:
        seq_path = os.path.join(sequences_path, seq)
        
        # Process both color and grayscale images if they exist
        for img_folder_name in ["image_2", "image_0"]:
            image_folder = os.path.join(seq_path, img_folder_name)
            
            if os.path.exists(image_folder):
                print(f"\nProcessing {img_folder_name} in sequence {seq}...")
                
                # Create output folder
                output_folder = os.path.join(seq_path, f"{img_folder_name}_undistorted")
                os.makedirs(output_folder, exist_ok=True)
                
                # Undistort images
                undistort_images(
                    image_folder, 
                    output_folder, 
                    camera_params['focal_length'],
                    camera_params['cx'],
                    camera_params['cy'],
                    camera_params['k'],
                    rename_sequentially
                )

def main():
    parser = argparse.ArgumentParser(description='Undistort images using Simple Radial camera model parameters')
    parser.add_argument('input_path', help='Path to input folder containing images or KITTI dataset')
    parser.add_argument('--output', help='Path to output folder for undistorted images (optional)')
    parser.add_argument('--cameras-file', help='Path to COLMAP cameras.txt file with SIMPLE_RADIAL parameters')
    parser.add_argument('--focal-length', type=float, help='Focal length parameter f')
    parser.add_argument('--cx', type=float, help='Principal point x coordinate')
    parser.add_argument('--cy', type=float, help='Principal point y coordinate')
    parser.add_argument('--k', type=float, help='Radial distortion parameter')
    parser.add_argument('--kitti', action='store_true', help='Process as KITTI dataset structure')
    parser.add_argument('--rename', action='store_true', help='Rename images sequentially (000000.png, 000001.png, etc.)')
    args = parser.parse_args()
    
    try:
        # Check if camera parameters are provided directly or via cameras file
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
        else:
            raise ValueError("Either provide a COLMAP cameras file with --cameras-file or specify all camera parameters (--focal-length, --cx, --cy, --k)")
        
        if args.kitti:
            if not args.cameras_file:
                raise ValueError("KITTI processing requires a cameras file (--cameras-file)")
            process_kitti_dataset(args.input_path, args.cameras_file, args.rename)
        else:
            if not args.output:
                raise ValueError("Output folder must be specified with --output for non-KITTI processing")
            undistort_images(args.input_path, args.output, focal_length, cx, cy, k, args.rename)
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
