import numpy as np
from PIL import Image
import open3d as o3d
import cv2
import plotly.graph_objects as go

def show_point_cloud(pcd, scailing=False, intensity=False, frame_ratio=0.8, coordinate_frame=True, cones=False):
    """_summary_

    Args:
        points (pcd): (N, 3)
        frame_ratio (float, optional): change the visualization frame Defaults to 0.2.
    """
    points = np.asarray(pcd.points)
    points_colors = np.asarray(pcd.colors)
    # valid_mask = np.isfinite(points).all(axis=1)
    # points = points[valid_mask]
    
    if scailing:
        mask = np.ones(points.shape[0])
        mask = np.logical_and(mask, points[:, 0] < 15)
        mask = np.logical_and(mask, points[:, 0] > 6)
        mask = np.logical_and(mask, points[:, 1] < 4)
        mask = np.logical_and(mask, points[:, 1] > -4)        
        mask = np.logical_and(mask, points[:, 2] > 0.2)
        points = points[mask, :]
        
    if intensity:
        colors = np.asarray(pcd.colors)
        intensity = colors[:, 0]
        intensity_8bit = (255 * (intensity - intensity.min()) / (intensity.ptp() + 1e-6)).astype(np.uint8)

        jet_colors = cv2.applyColorMap(intensity_8bit, cv2.COLORMAP_JET)[:, ::-1]
        jet_colors = jet_colors.astype(np.float64) / 255.0  
        jet_colors = jet_colors.reshape(-1, 3)
        print("Color shape:", jet_colors.shape, "dtype:", jet_colors.dtype)

        pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(jet_colors))
        
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=1,
                    color=points_colors)
    )])

    x_range = points[:, 0].max()*frame_ratio - points[:, 0].min()*frame_ratio
    y_range = points[:, 1].max()*frame_ratio - points[:, 1].min()*frame_ratio
    z_range = points[:, 2].max()*frame_ratio - points[:, 2].min()*frame_ratio

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='manual',
        aspectratio=dict(x=x_range, y=y_range, z=z_range)
    ))
    
    if coordinate_frame:
        # Length of the axes
        axis_length = 1

        # Create lines for the axes
        lines = [
            go.Scatter3d(x=[0, axis_length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red'), name='X-axis'),
            go.Scatter3d(x=[0, 0], y=[0, axis_length], z=[0, 0], mode='lines', line=dict(color='green'), name='Y-axis'),
            go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_length], mode='lines', line=dict(color='blue'), name='Z-axis')
        ]

        # Create cones (arrows) for the axes
        if cones:
            cones = [
                go.Cone(x=[axis_length], y=[0], z=[0], u=[axis_length], v=[0], w=[0], sizemode='absolute', sizeref=0.1, anchor='tail', showscale=False),
                go.Cone(x=[0], y=[axis_length], z=[0], u=[0], v=[axis_length], w=[0], sizemode='absolute', sizeref=0.1, anchor='tail', showscale=False),
                go.Cone(x=[0], y=[0], z=[axis_length], u=[0], v=[0], w=[axis_length], sizemode='absolute', sizeref=0.1, anchor='tail', showscale=False)
            ]
            for cone in cones:
                fig.add_trace(cone)

        # Add lines and cones to the figure
        for line in lines:
            fig.add_trace(line)


    fig.show()