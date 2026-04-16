import open3d as o3d
from utils.transformation import rotation_transform, xyz_rot_transform
import numpy as np
from utils.constants import *


def xyz1trans(matrix, points):
    """
    Apply the 4*4 matrix on each 3d point
    """
    points = points.copy()
    one = np.ones(list(points.shape[:-1])+[1])
    points = np.concatenate((points, one), -1)
    points = np.einsum('ij,bj->bi', matrix, points)
    points = points[:, :3]
    return points

def load_point_cloud(colors, depths, INTRINSIC, mask=None):
    xmap = np.arange(depths.shape[1])
    ymap = np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    fx, fy = INTRINSIC[0, 0], INTRINSIC[1, 1]
    cx, cy = INTRINSIC[0, 2], INTRINSIC[1, 2]
    points_z = depths.astype(np.float32)
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    points = np.stack([points_x, points_y, points_z],
                      axis=-1).astype(np.float32)
    depth_mask = (depths > 0.01)
    points = points[depth_mask]
    colors = colors[depth_mask]
    if mask is None:
        return points, colors
    else:
        return points, colors, mask[depth_mask]

def create_raw_point_cloud(colors, depths, cam_intrinsics, cam_to_d=None):
    """
    color, depth => point cloud
    no normalization
    no workspace filter
    """
    colors = np.array(colors).astype(np.float32) / 255.0
    depths = np.array(depths).astype(np.float32)
    # # imagenet normalization
    # colors = (colors - IMG_MEAN) / IMG_STD
    # create point cloud
    xmap = np.arange(depths.shape[1])
    ymap = np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]
    points_z = depths
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    points = np.stack([points_x, points_y, points_z], axis=-1)
    # filter invalid depths
    depth_mask = (depths > 0.01)
    points = points[depth_mask]
    colors = colors[depth_mask]
    # transform
    if cam_to_d is not None:
        one = np.ones(list(points.shape[:-1])+[1])
        points = np.concatenate((points, one), -1)
        points = np.einsum('ij,bj->bi', cam_to_d, points)
        points = points[:, :3]
    cloud = np.concatenate([points, colors], axis=-1)
    return cloud


def create_point_cloud(colors, depths, cam_intrinsics, cam_to_d=None, use_workspace=True):
    """
    color, depth => point cloud
    """
    colors = np.array(colors).astype(np.float32) / 255.0
    depths = np.array(depths).astype(np.float32)
    # imagenet normalization
    colors = (colors - IMG_MEAN) / IMG_STD
    # create point cloud
    xmap = np.arange(depths.shape[1])
    ymap = np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]
    points_z = depths
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    points = np.stack([points_x, points_y, points_z], axis=-1)
    # filter invalid depths
    depth_mask = (depths > 0.01) & (depths < 1)
    points = points[depth_mask]
    colors = colors[depth_mask]
    # transform
    if cam_to_d is not None:
        one = np.ones(list(points.shape[:-1])+[1])
        points = np.concatenate((points, one), -1)
        points = np.einsum('ij,bj->bi', cam_to_d, points)
        points = points[:, :3]
    # TODO
    # filter ouside workspace
    # x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
    # y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
    # z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
        if use_workspace:
            x_mask = ((points[:, 0] >= WORLD_WORKSPACE_MIN[0]) &
                      (points[:, 0] <= WORLD_WORKSPACE_MAX[0]))
            y_mask = ((points[:, 1] >= WORLD_WORKSPACE_MIN[1]) &
                      (points[:, 1] <= WORLD_WORKSPACE_MAX[1]))
            z_mask = ((points[:, 2] >= WORLD_WORKSPACE_MIN[2]) &
                      (points[:, 2] <= WORLD_WORKSPACE_MAX[2]))

            mask = (x_mask & y_mask & z_mask)
            points = points[mask]
            colors = colors[mask]
        else:
            mask = np.ones_like(points[:, 0], dtype=bool)
    else:
        mask = np.ones_like(points[:, 0], dtype=bool)
    # final cloud
    cloud = np.concatenate([points, colors], axis=-1)
    final_mask = np.zeros_like(depth_mask)
    final_mask[depth_mask] = mask
    return cloud, final_mask


def vis_actions(cloud, action, tcp_pose=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] * IMG_STD + IMG_MEAN)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)

    tcp_vis_list = []
    if tcp_pose is not None:
        tcp_vis_list.append(o3d.geometry.TriangleMesh.create_sphere(
            0.01).translate(tcp_pose[:3]))
    for raw_tcp in action:
        tcp_matrix = xyz_rot_transform(
            raw_tcp[:-1],
            from_rep='rotation_6d',
            to_rep="matrix",
        )
        tcp_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            0.05).transform(tcp_matrix)
        # tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[:3])
        tcp_vis_list.append(tcp_vis)
    o3d.visualization.draw_geometries([pcd, coordinate, *tcp_vis_list])
