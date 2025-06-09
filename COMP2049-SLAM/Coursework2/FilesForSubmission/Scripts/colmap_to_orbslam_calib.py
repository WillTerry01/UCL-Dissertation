#!/usr/bin/env python3

import os
import numpy as np
import argparse
from pathlib import Path
import cv2
import yaml

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
        
    # Skip header lines, find data line
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
    elif model == "OPENCV":
        # OPENCV format: fx, fy, cx, cy, k1, k2, p1, p2
        fx = float(params[4])
        fy = float(params[5])
        cx = float(params[6])
        cy = float(params[7])
        k1 = float(params[8]) if len(params) > 8 else 0.0
        k2 = float(params[9]) if len(params) > 9 else 0.0
        p1 = float(params[10]) if len(params) > 10 else 0.0
        p2 = float(params[11]) if len(params) > 11 else 0.0
        
        return {
            'model': "PINHOLE",  # Convert to PINHOLE for ORB-SLAM2
            'width': width,
            'height': height,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'dist_coeffs': np.array([k1, k2, p1, p2])
        }
    else:
        print(f"Warning: Camera model {model} not fully supported. Basic parameters extracted.")
        return {
            'model': model,
            'width': width,
            'height': height,
            'params': [float(p) for p in params[4:]]
        }

def read_opencv_yaml(yaml_file):
    """
    Read camera parameters from OpenCV YAML calibration file.
    
    Args:
        yaml_file (str): Path to OpenCV YAML calibration file
    
    Returns:
        dict: Camera parameters
    """
    with open(yaml_file, 'r') as f:
        calib_data = yaml.safe_load(f)
    
    # Extract camera matrix
    camera_matrix = np.array(calib_data.get('camera_matrix', {}).get('data', [])).reshape(3, 3)
    # Extract distortion coefficients
    dist_coeffs = np.array(calib_data.get('distortion_coefficients', {}).get('data', []))
    # Extract image dimensions
    width = calib_data.get('image_width', 0)
    height = calib_data.get('image_height', 0)
    
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    return {
        'model': 'PINHOLE',  # OpenCV calibration typically uses pinhole model
        'width': width,
        'height': height,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'dist_coeffs': dist_coeffs
    }

def read_opencv_xml(xml_file):
    """
    Read camera parameters from OpenCV XML calibration file.
    
    Args:
        xml_file (str): Path to OpenCV XML calibration file
    
    Returns:
        dict: Camera parameters
    """
    fs = cv2.FileStorage(xml_file, cv2.FILE_STORAGE_READ)
    
    # Extract camera matrix
    camera_matrix = fs.getNode('camera_matrix').mat()
    # Extract distortion coefficients
    dist_coeffs = fs.getNode('distortion_coefficients').mat()
    # Try to get image dimensions
    width = int(fs.getNode('image_width').real()) if fs.getNode('image_width').isReal() else 0
    height = int(fs.getNode('image_height').real()) if fs.getNode('image_height').isReal() else 0
    
    fs.release()
    
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    return {
        'model': 'PINHOLE',  # OpenCV calibration typically uses pinhole model
        'width': width,
        'height': height,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'dist_coeffs': dist_coeffs.flatten()
    }

def read_intrinsics_txt(intrinsics_file):
    """
    Read camera intrinsics from a txt file with the format:
    fx, fy, cx, cy, k1, k2, p1, p2
    
    Args:
        intrinsics_file (str): Path to intrinsics txt file
        
    Returns:
        dict: Camera parameters
    """
    with open(intrinsics_file, 'r') as f:
        line = f.readline().strip()
        params = [float(x) for x in line.split(',')]
    
    if len(params) != 8:
        raise ValueError(f"Expected 8 parameters (fx, fy, cx, cy, k1, k2, p1, p2), but got {len(params)}")
        
    fx, fy, cx, cy, k1, k2, p1, p2 = params
    
    return {
        'model': 'PINHOLE',
        'width': 0,  # These will be updated if provided via command line
        'height': 0,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'dist_coeffs': np.array([k1, k2, p1, p2])
    }

def resize_camera_parameters(camera_params, target_width, target_height):
    """
    Resize camera parameters based on target resolution.
    
    Args:
        camera_params (dict): Original camera parameters
        target_width (int): Target image width
        target_height (int): Target image height
        
    Returns:
        dict: Resized camera parameters
    """
    original_width = camera_params['width']
    original_height = camera_params['height']
    
    if original_width == 0 or original_height == 0:
        print("Warning: Original dimensions are not available, using target dimensions without scaling.")
        camera_params['width'] = target_width
        camera_params['height'] = target_height
        return camera_params
    
    # Calculate scaling factors
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    
    # Create a copy of parameters
    resized_params = camera_params.copy()
    
    # Update dimensions
    resized_params['width'] = target_width
    resized_params['height'] = target_height
    
    # Scale intrinsic parameters
    if 'fx' in camera_params:
        resized_params['fx'] = camera_params['fx'] * scale_x
        resized_params['fy'] = camera_params['fy'] * scale_y
        resized_params['cx'] = camera_params['cx'] * scale_x
        resized_params['cy'] = camera_params['cy'] * scale_y
    elif 'focal_length' in camera_params:
        # For SIMPLE_RADIAL model
        resized_params['focal_length'] = camera_params['focal_length'] * scale_x  # Using scale_x as general scale
        resized_params['cx'] = camera_params['cx'] * scale_x
        resized_params['cy'] = camera_params['cy'] * scale_y
    
    print(f"Camera parameters resized from {original_width}x{original_height} to {target_width}x{target_height}")
    return resized_params

def read_colmap_images(images_file):
    """
    Read image poses from COLMAP images.txt file.
    
    Args:
        images_file (str): Path to images.txt file from COLMAP
    
    Returns:
        dict: Dictionary of image information
    """
    images = {}
    with open(images_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                
                # Skip the next line with point correspondences
                _ = f.readline()
                
                # Store the image info
                images[image_id] = {
                    'id': image_id,
                    'qvec': qvec,
                    'tvec': tvec,
                    'camera_id': camera_id,
                    'name': image_name
                }
    return images

def qvec2rotmat(qvec):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        qvec (np.array): Quaternion [w, x, y, z]
    
    Returns:
        np.array: 3x3 rotation matrix
    """
    w, x, y, z = qvec
    R = np.zeros((3, 3))
    
    R[0, 0] = 1 - 2 * y**2 - 2 * z**2
    R[0, 1] = 2 * x * y - 2 * z * w
    R[0, 2] = 2 * x * z + 2 * y * w
    
    R[1, 0] = 2 * x * y + 2 * z * w
    R[1, 1] = 1 - 2 * x**2 - 2 * z**2
    R[1, 2] = 2 * y * z - 2 * x * w
    
    R[2, 0] = 2 * x * z - 2 * y * w
    R[2, 1] = 2 * y * z + 2 * x * w
    R[2, 2] = 1 - 2 * x**2 - 2 * y**2
    
    return R

def create_orbslam_calib(camera_params, output_file, images=None):
    """
    Create an ORB-SLAM2 style calibration file from camera parameters.
    
    Args:
        camera_params (dict): Camera parameters from COLMAP or OpenCV
        output_file (str): Path to save the calibration file
        images (dict, optional): Image information from COLMAP
    """
    
    # Create projection matrices
    if camera_params['model'] == "SIMPLE_RADIAL":
        f = camera_params['focal_length']
        cx = camera_params['cx']
        cy = camera_params['cy']
        
        # For KITTI-like format used by ORB-SLAM2, we create 4 projection matrices P0, P1, P2, P3
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
    
    # If we have image information, we can use the first image's pose to set up the coordinate system
    if images and len(images) > 0:
        # Get the first image (or a specific one if needed)
        first_image = list(images.values())[0]
        
        # Convert quaternion to rotation matrix
        R = qvec2rotmat(first_image['qvec'])
        t = first_image['tvec']
        
        # Create a 4x4 transformation matrix
        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = t
        
        # Create camera matrices (3x4)
        K0 = np.hstack((P0[:, :3], P0[:, 3:4]))
        K1 = np.hstack((P1[:, :3], P1[:, 3:4]))
        K2 = np.hstack((P2[:, :3], P2[:, 3:4]))
        K3 = np.hstack((P3[:, :3], P3[:, 3:4]))
        
        # Update the projection matrices with the camera pose
        P0 = K0 @ Rt
        P1 = K1 @ Rt
        P2 = K2 @ Rt
        P3 = K3 @ Rt
    
    # Write to file in KITTI format
    with open(output_file, 'w') as f:
        # Format each matrix as in KITTI
        f.write(f"P0: {' '.join([f'{val:.12e}' for val in P0.flatten()])}\n")
        f.write(f"P1: {' '.join([f'{val:.12e}' for val in P1.flatten()])}\n")
        f.write(f"P2: {' '.join([f'{val:.12e}' for val in P2.flatten()])}\n")
        f.write(f"P3: {' '.join([f'{val:.12e}' for val in P3.flatten()])}\n")
        
        # Add additional calibration info if available
        if 'dist_coeffs' in camera_params:
            dist = camera_params['dist_coeffs']
            if len(dist) >= 4:
                f.write(f"DistCoef: {dist[0]:.12e} {dist[1]:.12e} {dist[2]:.12e} {dist[3]:.12e}\n")
    
    print(f"Calibration file saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert COLMAP, OpenCV, or raw intrinsics to ORB-SLAM2 format')
    parser.add_argument('--colmap_cameras', help='Path to COLMAP cameras.txt file')
    parser.add_argument('--colmap_images', help='Path to COLMAP images.txt file to include pose information')
    parser.add_argument('--opencv_yaml', help='Path to OpenCV YAML calibration file')
    parser.add_argument('--opencv_xml', help='Path to OpenCV XML calibration file')
    parser.add_argument('--intrinsics_file', help='Path to txt file with intrinsics in format: fx,fy,cx,cy,k1,k2,p1,p2')
    parser.add_argument('--output', default='calib.txt', help='Output calibration file (default: calib.txt)')
    parser.add_argument('--image_width', type=int, help='Image width (if not specified in calibration file)')
    parser.add_argument('--image_height', type=int, help='Image height (if not specified in calibration file)')
    parser.add_argument('--resize', action='store_true', help='Resize camera parameters to match image_width and image_height')
    args = parser.parse_args()
    
    try:
        camera_params = None
        images = None
        
        # Read camera parameters from different possible sources
        if args.colmap_cameras:
            print(f"Reading COLMAP camera parameters from {args.colmap_cameras}")
            camera_params = read_colmap_camera(args.colmap_cameras)
            
            # Optionally read image poses
            if args.colmap_images and os.path.exists(args.colmap_images):
                print(f"Reading image poses from {args.colmap_images}")
                images = read_colmap_images(args.colmap_images)
                
        elif args.opencv_yaml:
            print(f"Reading OpenCV YAML camera parameters from {args.opencv_yaml}")
            camera_params = read_opencv_yaml(args.opencv_yaml)
            
        elif args.opencv_xml:
            print(f"Reading OpenCV XML camera parameters from {args.opencv_xml}")
            camera_params = read_opencv_xml(args.opencv_xml)
            
        elif args.intrinsics_file:
            print(f"Reading camera intrinsics from {args.intrinsics_file}")
            camera_params = read_intrinsics_txt(args.intrinsics_file)
            
        else:
            raise ValueError("No input calibration file specified. Use --colmap_cameras, --opencv_yaml, --opencv_xml, or --intrinsics_file")
        
        # Resize camera parameters if requested
        if args.resize and args.image_width and args.image_height:
            camera_params = resize_camera_parameters(camera_params, args.image_width, args.image_height)
        else:
            # If not resizing, just update dimensions if provided
            if args.image_width and camera_params:
                camera_params['width'] = args.image_width
            if args.image_height and camera_params:
                camera_params['height'] = args.image_height
            
        # Create calibration file
        create_orbslam_calib(camera_params, args.output, images)
        
        # Print summary of calibration parameters
        print("\nCalibration summary:")
        print(f"Image dimensions: {camera_params['width']}x{camera_params['height']}")
        if 'fx' in camera_params:
            print(f"Focal length: fx={camera_params['fx']:.6f}, fy={camera_params['fy']:.6f}")
            print(f"Principal point: cx={camera_params['cx']:.6f}, cy={camera_params['cy']:.6f}")
        elif 'focal_length' in camera_params:
            print(f"Focal length: f={camera_params['focal_length']:.6f}")
            print(f"Principal point: cx={camera_params['cx']:.6f}, cy={camera_params['cy']:.6f}")
            
        if 'dist_coeffs' in camera_params and len(camera_params['dist_coeffs']) >= 4:
            dist = camera_params['dist_coeffs']
            print(f"Distortion coefficients: k1={dist[0]:.6f}, k2={dist[1]:.6f}, p1={dist[2]:.6f}, p2={dist[3]:.6f}")
            
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main() 