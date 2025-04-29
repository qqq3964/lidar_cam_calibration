import traceback
import os
from datetime import datetime

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from tools.find_plane_equation_in_lidar_camera_coordinate import \
    calculate_plane_equation_in_lidar_camera_coordinate
from tools.read_calibration_file import read_yaml_file
from tools.optimizer import init_estimate_R_one_pose, init_estimate_t_one_pose, cost_function_one_pose_not_keep_R_orthogonal
from tools.visualizer import lidar_points_in_image, image_to_lidar_points
from tqdm import tqdm
import argparse

def optimize_rot_and_trans(
    calibration_data,
    camera_coordinate_plane_equation,
    lidar_plane_equation,
    lidar_points_on_plane,
    keep_rotation_matrix_orthogonal=True
):
    
    # initial estimate for R
    estimated_rotation_results = init_estimate_R_one_pose(
                                        lidar_plane_equation=lidar_plane_equation,
                                        camera_coordinate_plane_equation=camera_coordinate_plane_equation,
                                        )
    estimated_rotation_matrix = estimated_rotation_results['rotation_matrix']

    # initial estimate for t
    estimated_translation= init_estimate_t_one_pose(
        camera_coordinate_plane_equation=camera_coordinate_plane_equation,
        estimated_rotation_matrix=estimated_rotation_matrix, 
        lidar_plane_points=lidar_points_on_plane, 
        )

    # refine initial estimated R and t
    x0 = estimated_rotation_matrix.flatten().tolist() + estimated_translation.flatten().tolist() 
    fun = lambda x: cost_function_one_pose_not_keep_R_orthogonal(
        x=x,
        camera_coordinate_plane_equation=camera_coordinate_plane_equation,
        lidar_plane_points=lidar_points_on_plane,
    )
    
    result = least_squares(fun=fun, x0=x0, verbose=2)
    
    if keep_rotation_matrix_orthogonal == True:
        rotation_vec = result.x[0:3]
        rotation_matrix, _ = cv.Rodrigues(src=rotation_vec)
        rotation_matrix = np.reshape(rotation_matrix, newshape=(3,3))
        translation_vec = result.x[3:6]
        translation_vec = np.reshape(translation_vec, newshape=(3,1))
    else:
        rotation_matrix = result.x[0:9]
        rotation_matrix = np.reshape(rotation_matrix, newshape=(3,3))
        translation_vec = result.x[9:12]
        translation_vec = np.reshape(translation_vec, newshape=(3,1))

    return estimated_rotation_matrix, estimated_translation, rotation_matrix, translation_vec

def automatic_extrinsic_calibration(
    img_root,
    roi_pcd_root,
    whole_pcd_root,
    calibration_target_path,
    num_row,
    num_col,
    square,
    transform_matrix_path=None,
    ):
    
    # read image
    image_rgb_paths = sorted(os.listdir(img_root))
    roi_pcd_paths = sorted(os.listdir(roi_pcd_root))
    whole_pcd_paths = sorted(os.listdir(whole_pcd_root))
    
    plane_edges_equations_in_lidar_camera_coordinates = {}

    # calibration information related to camera
    calibration_data = read_yaml_file(path=calibration_target_path)
    print("Calibration Parameters:\n", calibration_data)
    plane_edges_equations_in_lidar_camera_coordinates = {
        'camera_coordinate_plane_equation': [],
        'lidar_plane_equation': [],
        'lidar_points_on_plane': []
        }
    for i in tqdm(range(len(image_rgb_paths))):
        try:
            img_bgr = cv.imread(os.path.join(img_root, image_rgb_paths[i]))
            rgb_image = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
            
            # point clouds
            point_cloud_target = np.load(os.path.join(roi_pcd_root, roi_pcd_paths[i]))
            point_cloud_target *= 1000
            
            point_cloud_scene = np.load(os.path.join(whole_pcd_root, whole_pcd_paths[i]))
            point_cloud_scene *= 1000

            ################################################################
            # remove distortion from camera with intrensic parameters of camera
            ################################################################    
            rgb_image = cv.undistort(
                src=rgb_image, 
                cameraMatrix=calibration_data['camera_matrix'],
                distCoeffs=calibration_data['distortion_coefficients']
                )

            ################################################################
            # Calculate plane equation and edges equations for calibration
            # target inside lidar and camera coordinate system
            ################################################################
            if transform_matrix_path is None:
                plane_edges_equations_in_lidar_camera_coordinate = calculate_plane_equation_in_lidar_camera_coordinate(
                                                                        point_cloud=point_cloud_target,
                                                                        calibration_data=calibration_data,
                                                                        rgb_image=rgb_image,
                                                                        num_row=num_row,
                                                                        num_col=num_col,
                                                                        square=square)
                for idx, (key, value) in enumerate(plane_edges_equations_in_lidar_camera_coordinate.items()):
                    plane_edges_equations_in_lidar_camera_coordinates[key].append(value)        
        except Exception:
            traceback.print_exc()

    ###################################################################
    #       Calculate R and t
    ###################################################################
    if transform_matrix_path is None:
        init_r, init_t, r, t = optimize_rot_and_trans(
            calibration_data=calibration_data,
            camera_coordinate_plane_equation=plane_edges_equations_in_lidar_camera_coordinates['camera_coordinate_plane_equation'],
            lidar_plane_equation=plane_edges_equations_in_lidar_camera_coordinates['lidar_plane_equation'],
            lidar_points_on_plane=plane_edges_equations_in_lidar_camera_coordinates['lidar_points_on_plane'],
            keep_rotation_matrix_orthogonal=False
        )

        print('=' * 30)
        print('Initial Estimated Rotation Matrix:')
        print(init_r)
        print('Initial Estimated Translation Matrix:')
        print(init_t)
        print('Rotation Matrix:')
        print(r)
        print('Translation Matrix:')
        print(t)
        print('=' * 30)

    if transform_matrix_path is not None:
        print('=========== Loaded an existing transform matrix. Rotation and translation will not be recalculated. ===========')
        init_r = np.load(os.path.join(transform_matrix_path, 'r.npy'))
        r = np.load(os.path.join(transform_matrix_path, 'r.npy'))
        init_t = np.load(os.path.join(transform_matrix_path, 't.npy'))
        t = np.load(os.path.join(transform_matrix_path, 't.npy'))
        
    # point clould points of calibrariotion target on image
    points_in_image, img_target_lidar_points = lidar_points_in_image(
        rgb_image=rgb_image,
        point_cloud=point_cloud_target,
        calibration_data=calibration_data,
        r_lidar_to_camera_coordinate=r,
        t_lidar_to_camera_coordinate=t
    )
    
    if whole_pcd_root is not None:
        # point clould points of whole scene on image
        points_in_image, img_scence_lidar_points = lidar_points_in_image(
            rgb_image=rgb_image,
            point_cloud=point_cloud_scene,
            calibration_data=calibration_data,
            r_lidar_to_camera_coordinate=r,
            t_lidar_to_camera_coordinate=t
        )
        
        # backprojection to lidar from image
        # image_to_lidar_points(
        #     rgb_image=rgb_image,
        #     point_cloud=point_cloud_scene,
        #     calibration_data=calibration_data,
        #     r_lidar_to_camera_coordinate=r,
        #     t_lidar_to_camera_coordinate=t
        # )
    else:
        img_scence_lidar_points = None


    if transform_matrix_path is None:
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")

        os.makedirs(os.path.join('results'), exist_ok=True)
        os.mkdir(os.path.join('results', dt_string))

        np.save(os.path.join('results', dt_string, 'init_r.npy'), init_r)
        np.save(os.path.join('results', dt_string, 'init_t.npy'), init_t)
        np.save(os.path.join('results', dt_string, 'r.npy'), r)
        np.save(os.path.join('results', dt_string, 't.npy'), t)
        np.save(os.path.join('results', dt_string, 'k.npy'), calibration_data['camera_matrix'])
        np.save(os.path.join('results', dt_string, 'distortion.npy'), calibration_data['distortion_coefficients'])
        np.savetxt(os.path.join('results', dt_string, 'init_r.txt'), init_r)
        np.savetxt(os.path.join('results', dt_string, 'init_t.txt'), init_t)
        np.savetxt(os.path.join('results', dt_string, 'r.txt'), r)
        np.savetxt(os.path.join('results', dt_string, 't.txt'), t)
    
        im_temp = Image.fromarray(img_target_lidar_points)
        im_temp.save(os.path.join('results', dt_string, "img_target_lidar_points.png"))

        if img_scence_lidar_points is not None:
            im_temp = Image.fromarray(img_scence_lidar_points)
            im_temp.save(os.path.join('results', dt_string, "img_scence_lidar_points.png"))

    return {'initial_r': init_r, 'initial_t': init_t, 'r': r, 't': t,
            'input_image': img_bgr,
            'target_lidar_points_projected_image': img_target_lidar_points,
            'scene_lidar_points_projected_image': img_scence_lidar_points}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='information')
    parser.add_argument('--img_root', type=str)
    parser.add_argument('--roi_pcd_root', type=str)
    parser.add_argument('--whole_pcd_root', type=str)
    parser.add_argument('--calibration_target_path', type=str)
    parser.add_argument('--num_row', type=int)
    parser.add_argument('--num_col', type=int)
    parser.add_argument('--square', type=int)
    parser.add_argument('--transform_matrix_path', type=str, default=None)
    
    args = parser.parse_args()

    all_output = automatic_extrinsic_calibration(
        img_root=args.img_root,
        roi_pcd_root=args.roi_pcd_root,
        whole_pcd_root=args.whole_pcd_root,
        calibration_target_path=args.calibration_target_path,
        num_row=args.num_row,
        num_col=args.num_col,
        square=args.square,
        transform_matrix_path=args.transform_matrix_path
    )

    init_r = all_output['initial_r']
    init_t = all_output['initial_t']
    r = all_output['r']
    t = all_output['t']


    print('=' * 30)
    print('Initial Estimated Rotation Matrix:')
    print(init_r)
    print('Initial Estimated Translation Matrix:')
    print(init_t)
    print('Rotation Matrix:')
    print(r)
    print('Translation Matrix:')
    print(t)
    print('=' * 30)

    plt.figure()
    plt.imshow(all_output['target_lidar_points_projected_image'])

    plt.figure()
    plt.imshow(all_output['scene_lidar_points_projected_image'])

    plt.show()

