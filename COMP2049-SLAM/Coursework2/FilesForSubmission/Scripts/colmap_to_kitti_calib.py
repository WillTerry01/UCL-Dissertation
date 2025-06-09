#!/usr/bin/env python3

import os
import numpy as np
import argparse

def read_colmap_camera(cameras_file):
    """
    Read camera parameters from COLMAP cameras.txt file.
    
    Args:
        cameras_file (str): Path to cameras.txt file from COLMAP
    
    Returns:
        dict: Camera parameters
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
    elif model == "PINHOLE":
        # PINHOLE format: fx, fy, cx, cy
        fx = float(params[4])
        fy = float(params[5])
        cx = float(params[6])
        cy = float(params[7])
        
        return {
            'model': model,
            'width': width,
            'height': height,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }
    else:
        print(f"Warning: Camera model {model} not fully supported. Basic parameters extracted.")
        return {
            'model': model,
            'width': width,
            'height': height,
            'params': [float(p) for p in params[4:]]
        }

def create_kitti_calib(camera_params, output_file):
    """
    Create a KITTI-style calibration file from camera parameters.
    
    Args:
        camera_params (dict): Camera parameters from COLMAP
        output_file (str): Path to save the calibration file
    """
    
    # Create projection matrices
    if camera_params['model'] == "SIMPLE_RADIAL":
        f = camera_params['focal_length']
        cx = camera_params['cx']
        cy = camera_params['cy']
        
        # For KITTI, we create 4 projection matrices P0, P1, P2, P3
        # P0 (left grayscale camera)
        P0 = np.zeros((3, 4))
        P0[0, 0] = f
        P0[0, 2] = cx
        P0[1, 1] = f
        P0[1, 2] = cy
        P0[2, 2] = 1.0
        
        # P1 (right grayscale camera) - simulated with baseline
        P1 = P0.copy()
        P1[0, 3] = -0.54 * f  # Simulated baseline of 54cm
        
        # P2 (left color camera) - same as P0 with small offset
        P2 = P0.copy()
        P2[0, 3] = 0.06 * f  # Small x-offset
        P2[1, 3] = -0.00016 * f  # Small y-offset
        P2[2, 3] = 0.000005  # Small z-offset
        
        # P3 (right color camera) - similar to P1 with offset
        P3 = P0.copy()
        P3[0, 3] = -0.47 * f  # Adjusted baseline
        P3[1, 3] = 0.0033 * f  # Small y-offset
        P3[2, 3] = 0.000007  # Small z-offset
    
    elif camera_params['model'] == "PINHOLE":
        fx = camera_params['fx']
        fy = camera_params['fy']
        cx = camera_params['cx']
        cy = camera_params['cy']
        
        # P0 (left grayscale camera)
        P0 = np.zeros((3, 4))
        P0[0, 0] = fx
        P0[0, 2] = cx
        P0[1, 1] = fy
        P0[1, 2] = cy
        P0[2, 2] = 1.0
        
        # P1, P2, P3 similar to above
        P1 = P0.copy()
        P1[0, 3] = -0.54 * fx
        
        P2 = P0.copy()
        P2[0, 3] = 0.06 * fx
        P2[1, 3] = -0.00016 * fy
        P2[2, 3] = 0.000005
        
        P3 = P0.copy()
        P3[0, 3] = -0.47 * fx
        P3[1, 3] = 0.0033 * fy
        P3[2, 3] = 0.000007
    
    else:
        # Default case - use a standard projection matrix
        print(f"Warning: Using default values for camera model {camera_params['model']}")
        # Assuming first parameter is focal length
        f = camera_params['params'][0]
        cx = camera_params['width'] / 2
        cy = camera_params['height'] / 2
        
        P0 = np.zeros((3, 4))
        P0[0, 0] = f
        P0[0, 2] = cx
        P0[1, 1] = f
        P0[1, 2] = cy
        P0[2, 2] = 1.0
        
        P1 = P0.copy()
        P1[0, 3] = -0.54 * f
        
        P2 = P0.copy()
        P2[0, 3] = 0.06 * f
        P2[1, 3] = -0.00016 * f
        P2[2, 3] = 0.000005
        
        P3 = P0.copy()
        P3[0, 3] = -0.47 * f
        P3[1, 3] = 0.0033 * f
        P3[2, 3] = 0.000007
    
    # Write to file in KITTI format
    with open(output_file, 'w') as f:
        # Format each matrix as in KITTI
        f.write(f"P0: {' '.join([f'{val:.12e}' for val in P0.flatten()])}\n")
        f.write(f"P1: {' '.join([f'{val:.12e}' for val in P1.flatten()])}\n")
        f.write(f"P2: {' '.join([f'{val:.12e}' for val in P2.flatten()])}\n")
        f.write(f"P3: {' '.join([f'{val:.12e}' for val in P3.flatten()])}\n")
    
    print(f"Calibration file saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert COLMAP camera parameters to KITTI calibration format')
    parser.add_argument('cameras_file', help='Path to COLMAP cameras.txt file')
    parser.add_argument('--output', default='calib.txt', help='Output calibration file (default: calib.txt)')
    args = parser.parse_args()
    
    try:
        camera_params = read_colmap_camera(args.cameras_file)
        create_kitti_calib(camera_params, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main() 