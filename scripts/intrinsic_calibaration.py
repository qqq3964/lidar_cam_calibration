import cv2
import numpy as np
import glob
import os
import random

# === Checkerboard settings ===
CHECKERBOARD = (6, 5)  # number of internal corners per chessboard row and column
SQUARE_SIZE = 0.15       # size of a square (e.g. in meters)

# Prepare 3D object points in the checkerboard's coordinate space
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# === Load all calibration images ===
image_dir = "/data/velodyne/Image"  # Folder containing chessboard images
all_images = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
images = random.sample(all_images, 100)

if not images:
    print("No images found in", image_dir)
    exit(1)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
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
    undistorted = cv2.undistort(img, cameraMatrix, -distCoeffs, None)
    concat = np.hstack((img, undistorted))
    cv2.imshow(f"Distortion {i+1} (Original | Undistorted)", concat)
    
    print(f"Showing image {i+1}/5. Press any key to continue...")
    key = cv2.waitKey(0)  # waits indefinitely for a key press
    if key == 27:  # ESC key to exit early
        print("Stopped by user.")
        break

cv2.destroyAllWindows()

# === Save calibration to file ===
np.savez("camera_calib_result.npz", K=cameraMatrix, D=distCoeffs, rvecs=rvecs, tvecs=tvecs)
print("\nCalibration saved to camera_calib_result.npz")
