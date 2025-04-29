import cv2
import numpy as np
import glob
import os

import yaml

# custom YAML dumper to force inline lists for "data" fields
class InlineListDumper(yaml.SafeDumper):
    pass

def represent_inline_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

# register this behavior for all Python lists
InlineListDumper.add_representer(list, represent_inline_list)

def save_ros_yaml(cameraMatrix, distCoeffs, image_size, save_path="configs/example.yaml"):
    fx = float(cameraMatrix[0, 0])
    fy = float(cameraMatrix[1, 1])
    cx = float(cameraMatrix[0, 2])
    cy = float(cameraMatrix[1, 2])

    ros_yaml = {
        "camera_name": "my_camera",
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [
                round(fx, 8), 0.0, round(cx, 8),
                0.0, round(fy, 8), round(cy, 8),
                0.0, 0.0, 1.0
            ]
        },
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": 5,
            "data": [round(float(x), 8) for x in distCoeffs.flatten()]
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ]
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [
                round(fx, 8), 0.0, round(cx, 8), 0.0,
                0.0, round(fy, 8), round(cy, 8), 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
        }
    }

    with open(save_path, 'w') as f:
        yaml.dump(ros_yaml, f, Dumper=InlineListDumper, default_flow_style=False, sort_keys=False)

    print(f"Saved to {save_path} in ROS camera_info format.")

def run(images,
        checkerboard):
    if not images:
        print("No images found in", image_dir)
        exit(1)
    # Prepare 3D object points in the checkerboard's coordinate space
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
            cv2.imshow('Corners', img)
            cv2.waitKey(10)
    cv2.destroyAllWindows()
    # === Perform calibration ===
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    # === Print results ===
    print("Camera matrix (K):")
    print(cameraMatrix)
    print("\nDistortion coefficients (D):")
    print(distCoeffs.ravel())
    print(f"\nHeight : {img.shape[0]}, Width : {img.shape[1]}")
    for i in range(5):
        img = cv2.imread(images[i])
        undistorted = cv2.undistort(img, cameraMatrix, distCoeffs, None)
        concat = np.hstack((img, undistorted))
        cv2.imshow(f"Distortion {i+1} (Original | Undistorted)", concat)
        print(f"Showing image {i+1}/5. Press any key to continue...")
        key = cv2.waitKey(0)  # waits indefinitely for a key press
        if key == 27:  # ESC key to exit early
            print("Stopped by user.")
            break
        
    save_ros_yaml(cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, image_size=(img.shape[0], img.shape[1]))
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    # === Checkerboard settings ===
    CHECKERBOARD = (6, 5)  # number of internal corners per chessboard row and column
    SQUARE_SIZE = 0.15       # size of a square (e.g. in meters)
    # === Load all calibration images ===
    image_dir = "example/Image"  # Folder containing chessboard images
    all_images = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
    print(all_images)
    images = all_images
    run(images=images,
        checkerboard=CHECKERBOARD)