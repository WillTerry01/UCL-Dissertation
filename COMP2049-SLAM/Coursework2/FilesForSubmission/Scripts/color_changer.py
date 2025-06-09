#!/usr/bin/env python3

import os
import cv2
import argparse
import glob
from pathlib import Path
import shutil
import re

def convert_to_grayscale(input_folder, output_folder=None, rename_sequentially=False):
    """
    Convert all RGB images in the input folder to grayscale and save them in the output folder.
    
    Args:
        input_folder (str): Path to the folder containing RGB images
        output_folder (str): Path to save grayscale images. If None, will create a new folder next to input_folder
        rename_sequentially (bool): If True, rename images as 000000.png, 000001.png, etc.
    
    Returns:
        str: Path to the output folder
    """
    # Get absolute path of input folder
    input_folder = os.path.abspath(input_folder)
    
    # If output folder is not specified, create one next to the input folder
    if output_folder is None:
        parent_folder = os.path.dirname(input_folder)
        folder_name = os.path.basename(input_folder)
        output_folder = os.path.join(parent_folder, folder_name + "_gray")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get a list of image files in the input folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    # Sort the files naturally - this helps when we want to rename sequentially
    # Extract numbers from filenames for natural sorting
    def natural_sort_key(s):
        # Extract numbers from the filename
        numbers = re.findall(r'\d+', os.path.basename(s))
        if numbers:
            return int(numbers[0])
        return os.path.basename(s)
    
    # Try natural sorting, fall back to regular sorting if it fails
    try:
        image_files.sort(key=natural_sort_key)
    except:
        image_files.sort()
    
    print(f"Found {len(image_files)} images in {input_folder}")
    
    # Process each image
    for i, img_path in enumerate(image_files):
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
                
            # Convert to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Create output path with either sequential name or original name
            if rename_sequentially:
                filename = f"{i:06d}.png"  # Format: 000000.png, 000001.png, etc.
            else:
                filename = os.path.basename(img_path)
                
            output_path = os.path.join(output_folder, filename)
            
            # Save grayscale image
            cv2.imwrite(output_path, gray_img)
            
            # Print progress
            if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                print(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Conversion completed. Grayscale images saved to {output_folder}")
    return output_folder

def process_kitti_dataset(dataset_path, rename_sequentially=False):
    """
    Process a KITTI-format dataset, converting all image sequences to grayscale.
    
    Args:
        dataset_path (str): Path to the KITTI dataset (containing sequences folder)
        rename_sequentially (bool): If True, rename images as 000000.png, 000001.png, etc.
    """
    sequences_path = os.path.join(dataset_path, "sequences")
    if not os.path.exists(sequences_path):
        raise ValueError(f"Sequences folder not found at {sequences_path}")
    
    # Find all sequence folders
    sequence_folders = [f for f in os.listdir(sequences_path) if os.path.isdir(os.path.join(sequences_path, f))]
    print(f"Found {len(sequence_folders)} sequence folders")
    
    for seq in sequence_folders:
        seq_path = os.path.join(sequences_path, seq)
        image_folder = os.path.join(seq_path, "image_2")  # KITTI RGB images are in image_2
        
        if os.path.exists(image_folder):
            print(f"\nProcessing sequence {seq}...")
            gray_folder = os.path.join(seq_path, "image_0")  # KITTI grayscale images convention
            
            # Create image_0 folder if it doesn't exist
            os.makedirs(gray_folder, exist_ok=True)
            
            # Convert images
            convert_to_grayscale(image_folder, gray_folder, rename_sequentially)
            
            # Copy calibration and other files if they exist
            for file in ["calib.txt", "times.txt"]:
                src_file = os.path.join(seq_path, file)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, os.path.join(gray_folder, ".."))
        else:
            print(f"Warning: No image_2 folder found in sequence {seq}")

def main():
    parser = argparse.ArgumentParser(description='Convert RGB images to grayscale')
    parser.add_argument('input_path', help='Path to input folder containing RGB images or KITTI dataset')
    parser.add_argument('--output', help='Path to output folder for grayscale images (optional)')
    parser.add_argument('--kitti', action='store_true', help='Process as KITTI dataset structure')
    parser.add_argument('--rename', action='store_true', help='Rename images sequentially (000000.png, 000001.png, etc.)')
    args = parser.parse_args()
    
    try:
        if args.kitti:
            process_kitti_dataset(args.input_path, args.rename)
        else:
            convert_to_grayscale(args.input_path, args.output, args.rename)
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
