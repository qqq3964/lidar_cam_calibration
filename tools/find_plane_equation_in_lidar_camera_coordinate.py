import matplotlib.pyplot as plt
from tools import camera_coordinate_plane_equation, lidar_coordinate_plane_equation

def calculate_plane_equation_in_lidar_camera_coordinate(
    point_cloud,
    calibration_data,
    rgb_image,
    num_row,
    num_col,
    square):
    
    # plane equation of calibration target in camera coordinate system
    image_plane_equation = camera_coordinate_plane_equation(
                            rgb_img=rgb_image,
                            num_row=num_row,
                            num_col=num_col,
                            square=square,
                            camera_matrix=calibration_data['camera_matrix'],
                            distortion_coefficients=calibration_data['distortion_coefficients'])
    
    lidar_plane_equation = lidar_coordinate_plane_equation(
        lidar_point_cloud=point_cloud)

    return {'camera_coordinate_plane_equation': image_plane_equation,
            'lidar_plane_equation': lidar_plane_equation['plane_equation'],
            'lidar_points_on_plane': lidar_plane_equation['lidar_points']}