from tkinter.messagebox import NO

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def find_corners_on_calibration_target(img, num_row, num_col, square):
    """
    img: image that contain calibration target (RGB image)
    num_row: number of inside corners in row direction
    num_col: number of inside corners in col direction
    square: len of each calibration target square in mm

    returned points are sorted from left to right and top to bottom, also return results is in opencv format (x, y, z),
    x is in direction of horizon and y in direction of vertical axis
    """
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((num_row*num_col,3), np.float32)
    objp[:,:2] = np.mgrid[0:num_col,0:num_row].T.reshape(-1,2)
    objp *= square

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (num_col, num_row))

    # If found, add object points, image points (after refining them)
    if ret == True:
        # corners in subpixel space
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        img_copy = np.copy(img)
        cv.drawChessboardCorners(img_copy, (num_col, num_row), corners2, ret)
        
        corners = np.reshape(corners, newshape=(corners.shape[0], 2))
        corners2 = np.reshape(corners2, newshape=(corners2.shape[0], 2))
     
        return {'points_in_3D': objp, 'points_in_image': corners, 'points_in_image_sub_pixel': corners2, 'image_corners': img_copy}
    else:
        return None


def calculate_plane_equation_by_three_points(three_points):
    """
    The input is a 3 * 3 numpu array, each row is a point.
    It returns plane equation that passes those points: a numpy array with shape (4, )
    """
    # calculate plane equition ax+by+cz+d = 0
    vec_1 = three_points[1, :] - three_points[0, :]
    vec_2 = three_points[2, :] - three_points[0, :] 
    normal = np.cross(vec_1, vec_2)
    if normal[2] < 0:
        normal *= -1
    normal /= np.linalg.norm(normal)
    d = -1 * (normal[0] * three_points[0, 0] + normal[1] * three_points[0, 1] + normal[2] * three_points[0, 2])
    plane_eqiotion = np.array([normal[0], normal[1], normal[2], d])

    return plane_eqiotion

def rotation_and_translation_of_target_in_camera_image(object_points, image_points, camera_matrix, distortion_coefficients):
    """
    object_points: Array of object points in the object coordinate space
    image_points: Array of corresponding image points
    camera_matrix: Input camera intrinsic matrix (3 in 3)
    distortion_coefficients: Input vector of distortion coefficients



    Output (rvec and tvec), rotation vector that, together with translation vector, brings points from the model coordinate system
    to the camera coordinate system. 
    rvec is in radian
    tvec is in mm
    """
    # retval, rvec, tvec = cv.solvePnP(objectPoints=object_points, imagePoints=image_points, cameraMatrix=camera_matrix, distCoeffs=distortion_coefficients)
    retval, rvec, tvec = cv.solvePnP(objectPoints=object_points, imagePoints=image_points, cameraMatrix=camera_matrix, distCoeffs=None)


    if retval == True:
        return {'rotation_vector': rvec, 'translation_vector': tvec}
    else:
        return None

def get_calibration_target_plane_equation_in_image(object_points, image_points, camera_matrix, distortion_coefficients):
    """
    Find plane equation of a calibration target inside and image. The plane equation is in camera coordinate system and in 
    homogenous format
    """

    rvec_tvec = rotation_and_translation_of_target_in_camera_image(object_points=object_points,
                                                       image_points=image_points,
                                                       camera_matrix=camera_matrix,
                                                       distortion_coefficients=distortion_coefficients)

    if rvec_tvec is None:
        return None
    
    # rotation vector (numpy array 3 * 1)
    rvec = rvec_tvec['rotation_vector']
    # translation vector (numpy array 3 * 1)
    tvec = rvec_tvec['translation_vector']

    # convert rotation vector to rotation matrix
    rotation_matrix, _ = cv.Rodrigues(src=rvec)

    # three points on calibration target obejct
    four_points = np.array([[0, 0, 0], [1000, 0, 0], [1000, 1000, 0], [0, 1000, 0]])

    # rotate and translate points from object to camera coordinate
    four_points = np.dot(rotation_matrix, four_points.T) + tvec
    four_points = four_points.T

    # calculate plane equation
    plane_equation = calculate_plane_equation_by_three_points(three_points=four_points[0:3, :])

    return plane_equation


def camera_coordinate_plane_equation(rgb_img, num_row, num_col, square, camera_matrix, distortion_coefficients):
    """
    Find plane equation of a calibration target inside and image. The plane equation is in camera coordinate system and in 
    homogenous format
    """
    
    # find corners on calibration target
    points_3d_image_image_subpix = find_corners_on_calibration_target(
                                        img=rgb_img,
                                        num_row=num_row,
                                        num_col=num_col,
                                        square=square)

    if points_3d_image_image_subpix is None:
        raise ValueError('Can not find corners on checkerboard.')

    # find plane equation of calibtarion target inside image
    plane_equation = get_calibration_target_plane_equation_in_image(
                                object_points=points_3d_image_image_subpix['points_in_3D'],
                                image_points=points_3d_image_image_subpix['points_in_image_sub_pixel'],
                                camera_matrix=camera_matrix,
                                distortion_coefficients=distortion_coefficients
                            )
    
    return plane_equation

def calculate_plane_equation(three_points):
    # calculate plane equition ax+by+cz+d = 0
    vec_1 = three_points[1, :] - three_points[0, :]
    vec_2 = three_points[2, :] - three_points[0, :] 
    normal = np.cross(vec_1, vec_2)
    if normal[0] < 0:
        normal *= -1
    normal /= np.linalg.norm(normal)
    d = -1 * (normal[0] * three_points[0, 0] + normal[1] * three_points[0, 1] + normal[2] * three_points[0, 2])
    plane_eqiotion = np.array([normal[0], normal[1], normal[2], d])

    return plane_eqiotion

def distance_of_points_to_plane(point_cloud, plane_eqiotion):
    # distance of points in point cloud to the plane
    point_cloud_with_one = np.hstack((point_cloud, np.ones(shape=(point_cloud.shape[0], 1))))
    distance_points_to_plane = np.abs(np.dot(point_cloud_with_one, plane_eqiotion.T))

    return distance_points_to_plane

def find_inliers(distance_points_to_plane, distance_to_be_inlier):
    # find inliers
    inliers_index = np.argwhere(distance_points_to_plane <= distance_to_be_inlier)
    inliers_index = np.reshape(inliers_index, newshape=(-1))

    return inliers_index

def ransac_plane_in_lidar(lidar_point, maximum_iteration=5000, inlier_ratio=0.9, distance_to_be_inlier=10):
    """
    lidar_point: numpy array with shape of (n, 3), all measurements are in mm.
    maximum_iteration: maximum iteration before halting the program.
    inlier_ratio: it will stop algorithm if the 90% or more of data in point cloud considered as inliers. 
    distance_to_be_inlier: if a point has a distance equal or less than this value, it will considered as inliers.
    """
    
    point_cloud_orginal = np.copy(lidar_point)
    
    best_ratio_plane = [0, None]
    
    # centroid of points in point cloud
    plane_centroid = np.mean(point_cloud_orginal, axis=0)

    for _ in range(maximum_iteration):
        
        # randomly select three points
        three_index = np.random.choice([idx for idx in range(point_cloud_orginal.shape[0])], size=3, replace=False)
        three_points = point_cloud_orginal[three_index]

        # calculate plane equation ax+by+cz+d = 0
        plane_eqiotion = calculate_plane_equation(three_points=three_points)

        # distance of points in point cloud to the plane
        distance_points_to_plane_all_set = distance_of_points_to_plane(point_cloud=point_cloud_orginal, plane_eqiotion=plane_eqiotion)

        # find inliers
        inliers_index_all_set = find_inliers(distance_points_to_plane=distance_points_to_plane_all_set, distance_to_be_inlier=distance_to_be_inlier)

        # find inliers ratio
        inlier_to_all_points_all_set = inliers_index_all_set.shape[0]/distance_points_to_plane_all_set.shape[0]

        if inlier_to_all_points_all_set > best_ratio_plane[0]:
            best_ratio_plane[0] = inlier_to_all_points_all_set
            best_ratio_plane[1] = plane_eqiotion

            if inlier_ratio <= inlier_to_all_points_all_set:
                break

    return {'inlier_to_all_data_ratio':best_ratio_plane[0], 'plane_equation':best_ratio_plane[1], 'plane_centroid': plane_centroid}

def lidar_coordinate_plane_equation(lidar_point_cloud):
    # find plane equation
    best_ratio_plane = ransac_plane_in_lidar(lidar_point=lidar_point_cloud)

    description = 'plane equation: ax+by+cz+d=0, each line equation: p0 a point on line and t the direction vector'
    return {'plane_equation': best_ratio_plane['plane_equation'],
            'lidar_points': lidar_point_cloud}