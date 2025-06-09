import numpy as np
from scipy.spatial.transform import Rotation as R

COLMAP_IMAGES_TXT_PATH = "/Users/willterry/UCL/Term2/COMP0249/COMP0249_24-25_ORB_SLAM2/TrackColmap/0txt/images.txt"
OUTPUT_PATH = "/Users/willterry/UCL/Term2/COMP0249/COMP0249_24-25_ORB_SLAM2/Datasets/OWN_results/CTT_Track.txt"

def convert_colmap_to_tum(images_txt_path, output_path, fps=30):
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()

    tum_poses = []
    timestamp = 0.0
    dt = 1.0 / fps

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("#") or line == "":
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            # Parse pose line
            image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, img_name = parts[:10]
            qw, qx, qy, qz = map(float, [qw, qx, qy, qz])
            tx, ty, tz = map(float, [tx, ty, tz])

            # COLMAP gives world-to-camera rotation (R_cw) and translation (t_cw)
            # Step 1: Convert quaternion to rotation matrix
            r = R.from_quat([qx, qy, qz, qw])  # (x, y, z, w)
            R_cw = r.as_matrix()

            # Step 2: Invert to camera-to-world
            R_wc = R_cw.T
            t_wc = -R_wc @ np.array([tx, ty, tz])

            # Step 3: Apply coordinate frame transformation (flip Y and Z)
            T = np.diag([1, -1, -1])
            R_wc_TUM = T @ R_wc
            t_wc_TUM = T @ t_wc

            # Step 4: Back to quaternion
            r_tum = R.from_matrix(R_wc_TUM)
            qx_tum, qy_tum, qz_tum, qw_tum = r_tum.as_quat()

            # Compose TUM line: timestamp tx ty tz qx qy qz qw
            tum_poses.append(f"{timestamp:.6f} {t_wc_TUM[0]} {t_wc_TUM[1]} {t_wc_TUM[2]} {qx_tum} {qy_tum} {qz_tum} {qw_tum}")

            timestamp += dt
            i += 2  # Skip the next line (2D points)
        else:
            i += 1  # In case something unexpected comes up

    with open(output_path, 'w') as out:
        out.write("\n".join(tum_poses))

    print(f"[✓] Converted COLMAP poses to TUM format at: {output_path}")

# Example usage
convert_colmap_to_tum(COLMAP_IMAGES_TXT_PATH, OUTPUT_PATH, fps=12)
