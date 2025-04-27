import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import io
import warnings

import cv2

warnings.filterwarnings("ignore")

def show_point_cloud(point_cloud, normal_vector=None, intersection_points=None, title=None, marker=None):
    
    if  isinstance(point_cloud, list):
        point_could_temp = []
        for sub_list in point_cloud:
            point_could_temp.append(sub_list)
    else:
        point_could_temp = np.vstack((point_cloud, point_cloud[0,:]))
    
    fig = plt.figure()
    plt.autoscale(False)
    ax = plt.axes(projection='3d')

    if  isinstance(point_cloud, list):
        for  sub_list in point_could_temp:
            ax.plot3D(sub_list[:, 0], sub_list[:, 1], sub_list[:, 2], label='calibration target')
    else:
        # plot calibration target
        if marker is None:
            ax.plot3D(point_could_temp[:, 0], point_could_temp[:, 1], point_could_temp[:, 2], 'gray', label='calibration target')
        else:
            ax.plot3D(point_could_temp[:, 0], point_could_temp[:, 1], point_could_temp[:, 2], 'gray', label='calibration target', marker=marker)

    # plot lidar
    ax.scatter3D(0, 0, 0, 'green', label='LiDAR')
    
    # plot normal vector
    if normal_vector is not None:
        middle = np.mean(point_cloud, axis=0)
        ax.scatter3D(middle[0], middle[1], middle[2], 'purple')
        ax.plot3D([middle[0], middle[0]+normal_vector[0]*100],
                  [middle[1], middle[1]+normal_vector[1]*100], [middle[2], middle[2]+normal_vector[2]*100],
                  'red', label='Normal Vector')

    if intersection_points is not None:
        ax.scatter3D(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], 'blue', label='intersecion points')


    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    if normal_vector is None:
        plt.title(title)
    else:
        if  isinstance(point_cloud, list):
            plt.title("{}\nNormal: {}, Num Lines".format(title, normal_vector, len(point_cloud)))
        else:
            plt.title("{}\nNormal: {}".format(title, normal_vector))

    plt.legend()

    numpy_img = get_img_from_fig(fig=fig)

    # close opend figure
    plt.close(fig)

    return numpy_img

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def lidar_points_in_image(
    rgb_image, 
    point_cloud,
    calibration_data,
    r_lidar_to_camera_coordinate,
    t_lidar_to_camera_coordinate, 
):

    # keep points that are in front of LiDAR
    points_in_front_of_lidar = []
    for point_i in point_cloud:
        if point_i[0] >= 0:
            points_in_front_of_lidar.append(point_i)
    point_cloud = np.array(points_in_front_of_lidar)
    
    # translate lidar points to camera coordinate system
    points_in_camera_coordinate = np.dot(r_lidar_to_camera_coordinate, point_cloud.T) + t_lidar_to_camera_coordinate

    # project points form camera coordinate to image
    points_in_image = np.dot(calibration_data['camera_matrix'], points_in_camera_coordinate)
    points_in_image = points_in_image / points_in_image[2, :]
    points_in_image = points_in_image[0:2, :]
    points_in_image = points_in_image.T
    
    # keep points that are inside image
    points_inside_image = []
    depth_inside_image = []
    
    for i, point_i in enumerate(points_in_image):
        if (0 <= point_i[0] < rgb_image.shape[1]) and (0 <= point_i[1] < rgb_image.shape[0]):
            points_inside_image.append(point_i)
            depth_inside_image.append(points_in_camera_coordinate[2, i])
    
    points_in_image = np.array(points_inside_image)
    depth_inside_image = np.array(depth_inside_image)
    
    fig = plt.figure()
    plt.imshow(rgb_image)
    plt.scatter(points_in_image[:, 0].tolist(), points_in_image[:, 1].tolist(), c=depth_inside_image, s=3)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    img_lidar_points = get_img_from_fig(fig=fig, dpi=500)

    return points_in_image, img_lidar_points
    
def image_to_lidar_points(
    rgb_image, 
    point_cloud,
    calibration_data,
    r_lidar_to_camera_coordinate,
    t_lidar_to_camera_coordinate, 
):
    # keep points that are in front of LiDAR
    points_in_front_of_lidar = []
    for point_i in point_cloud:
        if point_i[0] >= 0:
            points_in_front_of_lidar.append(point_i)
    point_cloud = np.array(points_in_front_of_lidar)
    
    # image to camera coordinate system
    points_in_camera_coordinate = np.dot(r_lidar_to_camera_coordinate, point_cloud.T) + t_lidar_to_camera_coordinate

    # project points form camera coordinate to image
    points_in_image = np.dot(calibration_data['camera_matrix'], points_in_camera_coordinate)
    points_in_image = points_in_image / points_in_image[2, :]
    points_in_image = points_in_image[0:2, :]
    points_in_image = points_in_image.T
    
    # keep points that are inside image
    points_inside_image = []
    depth_inside_image = []
    
    for i, point_i in enumerate(points_in_image):
        if (0 <= point_i[0] < rgb_image.shape[1]) and (0 <= point_i[1] < rgb_image.shape[0]):
            points_inside_image.append(point_i)
            depth_inside_image.append(points_in_camera_coordinate[2, i])
    
    points_in_image = np.array(points_inside_image)
    depth_inside_image = np.array(depth_inside_image)

    # image to lidar
    homo = np.ones((points_in_image.shape[0], 1))
    points_in_image_homo = np.concatenate([points_in_image, homo], axis=1)
    s = np.expand_dims(depth_inside_image, axis=0)
    points_in_camera_homo = np.linalg.inv(calibration_data['camera_matrix']) @ points_in_image_homo.T * s
    
    # rgb matched lidar points
    rows = points_in_image[:, 1].astype(int)
    cols = points_in_image[:, 0].astype(int)
    intensity = rgb_image[rows, cols] / 255.0
    
    points_in_camera_pcd = o3d.geometry.PointCloud()
    points_in_camera_pcd.points = o3d.utility.Vector3dVector(points_in_camera_homo.T)
    points_in_camera_pcd.colors = o3d.utility.Vector3dVector(intensity)
    
    o3d.visualization.draw_geometries([points_in_camera_pcd])