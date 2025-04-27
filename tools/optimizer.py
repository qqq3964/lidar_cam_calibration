import math
import numpy as np
import cv2 as cv
import numpy as np

def is_rotation_matrix(R) :
    """
    Checks if a matrix is a valid rotation matrix.
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotation_matrix_to_euler_angles(R):
    """
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
    """
    assert(is_rotation_matrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def init_estimate_R_one_pose(lidar_plane_equation, 
                             camera_coordinate_plane_equation):
    N = len(lidar_plane_equation)
    
    batch_m_l = []
    batch_m_c = []
    for i in range(N):
        n_l = lidar_plane_equation[i][0:3]
        batch_m_l.append(n_l)
        
        n_c = camera_coordinate_plane_equation[i][0:3]
        batch_m_c.append(n_c)

    batch_m_l = np.stack((batch_m_l)).T
    batch_m_c = np.stack((batch_m_c)).T
    m = np.dot(batch_m_l, batch_m_c.T)

    u, s, v_t = np.linalg.svd(a=m)

    r_estimate_matrix = np.dot(v_t.T, u.T)
    r_estimate_vector_radian = rotation_matrix_to_euler_angles(r_estimate_matrix)
    r_estimate_vector_degree = r_estimate_vector_radian * (180/np.math.pi)

    return {'rotation_matrix': r_estimate_matrix, 'rotation_vector_radian':r_estimate_vector_radian, 'rotation_vector_degree':r_estimate_vector_degree}

def calculate_A(line_direction):
    # direction of line
    line_direction = np.copy(line_direction)
    line_direction = np.reshape(line_direction, newshape=(-1, 1))
    if line_direction.shape[0] != 3:
        raise ValueError('The shape of direction vector for line equation is not correct.')

    matrix_a = np.identity(n=3) - np.dot(line_direction, line_direction.T)

    return matrix_a

def init_estimate_t_one_pose(camera_coordinate_plane_equation,
                             estimated_rotation_matrix, 
                             lidar_plane_points):

    N = len(camera_coordinate_plane_equation)
    
    batch_matrix_left = list()
    batch_matrix_right = list()
    for i in range(N):
        # normal vector and d of plane in camera coordinate (ax+by+cz+d=0)
        n_c = camera_coordinate_plane_equation[i][0:3]
        d_c = camera_coordinate_plane_equation[i][3]

        # convert matrixes to proper size
        n_c = np.reshape(n_c, newshape=(-1, 1))
        d_c = np.reshape(d_c, newshape=(-1, 1))
        lidar_plane_centroid_i = lidar_plane_points[i].T

        # create linear system, euqation Matrix_left * t= Vector_right 
        matrix_left = np.repeat(n_c.T, repeats=lidar_plane_centroid_i.shape[1], axis=0)
        matrix_right = -np.dot(n_c.T, np.dot(estimated_rotation_matrix, lidar_plane_centroid_i)) - d_c
        matrix_right = matrix_right.T
        batch_matrix_left.append(matrix_left)
        batch_matrix_right.append(matrix_right)
    batch_matrix_left = np.concatenate(batch_matrix_left, axis=0)
    batch_matrix_right = np.concatenate(batch_matrix_right, axis=0)
    
    # solve least squre problem
    estimated_t = np.dot(np.dot(np.linalg.inv(np.dot(batch_matrix_left.T, batch_matrix_left)), batch_matrix_left.T), batch_matrix_right)

    return estimated_t


def cost_function_one_pose_keep_R_orthogonal(
    x,
    camera_coordinate_plane_equation,
    camera_coordinate_edges_equation,
    lidar_plane_points,
    lidar_edges_points
):
    # rotation vector
    rotation_vector = x[0:3]
    # convert rotation vector to rotation matrix
    rotation_matrix, _ = cv.Rodrigues(src=rotation_vector)

    # translation vector
    translation_vector = x[3:6]
    translation_vector = np.reshape(translation_vector, newshape=(3, 1))

    # normal vector and d of plane in camera coordinate (ax+by+cz+d=0)
    n_c = camera_coordinate_plane_equation.tolist()[0:3]
    d_c = camera_coordinate_plane_equation.tolist()[3]
    
    # points on calibration target edges in camera coordinate
    p1_c =  camera_coordinate_edges_equation['left_lower_edge_equation'][0].tolist()
    p2_c =  camera_coordinate_edges_equation['left_upper_edge_equation'][0].tolist()
    p3_c =  camera_coordinate_edges_equation['right_upper_edge_equation'][0].tolist()
    p4_c =  camera_coordinate_edges_equation['right_lower_edge_equation'][0].tolist()

    # direction of calibration target edges in camera coordinate
    l1_c =  camera_coordinate_edges_equation['left_lower_edge_equation'][1].tolist()
    l2_c =  camera_coordinate_edges_equation['left_upper_edge_equation'][1].tolist()
    l3_c =  camera_coordinate_edges_equation['right_upper_edge_equation'][1].tolist()
    l4_c =  camera_coordinate_edges_equation['right_lower_edge_equation'][1].tolist()

    # A = I-d.d
    matrix_A_1 = calculate_A(line_direction=l1_c)
    matrix_A_2 = calculate_A(line_direction=l2_c)
    matrix_A_3 = calculate_A(line_direction=l3_c)
    matrix_A_4 = calculate_A(line_direction=l4_c)

    # convert matrixes to proper size
    n_c = np.reshape(n_c, newshape=(-1, 1))
    d_c = np.reshape(d_c, newshape=(-1, 1))
    p1_c = np.reshape(p1_c, newshape=(-1, 1))
    p2_c = np.reshape(p2_c, newshape=(-1, 1))
    p3_c = np.reshape(p3_c, newshape=(-1, 1))
    p4_c = np.reshape(p4_c, newshape=(-1, 1))
    lidar_plane_points = np.transpose(lidar_plane_points)
    lidar_l1_points = np.transpose(lidar_edges_points['left_lower_points'])
    lidar_l2_points = np.transpose(lidar_edges_points['left_upper_points'])
    lidar_l3_points = np.transpose(lidar_edges_points['right_upper_points'])
    lidar_l4_points = np.transpose(lidar_edges_points['right_lower_points'])

    # cost for points of calibration target
    cost_of_points_on_target = np.dot(n_c.T, np.dot(rotation_matrix, lidar_plane_points)+translation_vector)+d_c 
    cost_of_points_on_target = np.linalg.norm(cost_of_points_on_target, axis=0) ** 2
    cost_of_points_on_target = np.mean(cost_of_points_on_target)

    # cost for points of left lower edge of calibration target
    cost_of_points_on_edge_1 = np.dot(matrix_A_1, (np.dot(rotation_matrix, lidar_l1_points)-p1_c+translation_vector))
    cost_of_points_on_edge_1 = np.linalg.norm(cost_of_points_on_edge_1, axis=0) ** 2
    cost_of_points_on_edge_1 = np.mean(cost_of_points_on_edge_1)

    # cost for points of left upper edge of calibration target
    cost_of_points_on_edge_2 = np.dot(matrix_A_2, (np.dot(rotation_matrix, lidar_l2_points)-p2_c+translation_vector))
    cost_of_points_on_edge_2 = np.linalg.norm(cost_of_points_on_edge_2, axis=0) ** 2
    cost_of_points_on_edge_2 = np.mean(cost_of_points_on_edge_2)

    # cost for points of right upper edge of calibration target
    cost_of_points_on_edge_3 = np.dot(matrix_A_3, (np.dot(rotation_matrix, lidar_l3_points)-p3_c+translation_vector))
    cost_of_points_on_edge_3 = np.linalg.norm(cost_of_points_on_edge_3, axis=0) ** 2
    cost_of_points_on_edge_3 = np.mean(cost_of_points_on_edge_3)

    # cost for points of right lower edge of calibration target
    cost_of_points_on_edge_4 = np.dot(matrix_A_4, (np.dot(rotation_matrix, lidar_l4_points)-p4_c+translation_vector))
    cost_of_points_on_edge_4 = np.linalg.norm(cost_of_points_on_edge_4, axis=0) ** 2
    cost_of_points_on_edge_4 = np.mean(cost_of_points_on_edge_4)

    total_cost = cost_of_points_on_target + cost_of_points_on_edge_1 + cost_of_points_on_edge_2 + cost_of_points_on_edge_3 + cost_of_points_on_edge_4 

    return total_cost

def cost_function_one_pose_not_keep_R_orthogonal(
    x,
    camera_coordinate_plane_equation,
    lidar_plane_points,
):
    # rotation matrix and translation vector
    rotation_matrix = x[0:9]
    rotation_matrix = np.reshape(rotation_matrix, newshape=(3, 3))
    translation_vector = x[9:12]
    translation_vector = np.reshape(translation_vector, newshape=(3, 1))

    # total loss
    total_costs = 0.0
    N = len(camera_coordinate_plane_equation)
    for i in range(N):
        # normal vector and d of plane in camera coordinate (ax+by+cz+d=0)
        n_c = camera_coordinate_plane_equation[i][0:3]
        d_c = camera_coordinate_plane_equation[i][3]

        # convert matrixes to proper size
        n_c_ = np.reshape(n_c, newshape=(-1, 1))
        d_c_ = np.reshape(d_c, newshape=(-1, 1))
        lidar_plane_points_ = np.transpose(lidar_plane_points[i])

        # cost for points of calibration target
        cost_of_points_on_target = np.dot(n_c_.T, np.dot(rotation_matrix, lidar_plane_points_)+translation_vector)+d_c_ 
        cost_of_points_on_target = np.linalg.norm(cost_of_points_on_target, axis=0) ** 2
        cost_of_points_on_target = np.mean(cost_of_points_on_target)

        total_cost = cost_of_points_on_target 
        total_costs += total_cost
    total_costs = total_costs / N
    return total_costs