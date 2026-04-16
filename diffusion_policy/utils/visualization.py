from utils.tools import imshow
import matplotlib.pyplot as plt
from utils.constants import *
import time
import torch
import open3d as o3d
import cv2
from utils.transformation import rotation_transform, xyz_rot_transform


def get_color(i, n, to255=False):
    color1 = np.array((1., 0., 0.))
    color2 = np.array((0., 0., 1.))
    color = (color1*(n-i) + color2*i) / n
    color = list(color)
    if to255:
        color = [int(x*255) for x in color]
    return color


def get_colors(i, n, to255=False):
    color1 = np.array((1., 0., 0.)).reshape(3, 1)
    color2 = np.array((0., 0., 1.)).reshape(3, 1)
    color = (color1*(n-i) + color2*i) / n
    color = color.T
    if to255:
        color = (color * 255).astype(np.uint8)
    return color


def vis3d(cloud=None, point=[], normalize=True, pose=[], other_vis=[]):
    o = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
    if cloud is not None:
        if isinstance(cloud, torch.Tensor):
            cloud = cloud.cpu()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(
            cloud[:, 3:] * IMG_STD + IMG_MEAN if normalize else cloud[:, 3:])
    else:
        pcd = o3d.geometry.TriangleMesh.create_box(1, 1, 0.001).paint_uniform_color(
            (0.3, 0.6, 0.3)).translate((0, -0.5, 0))

    points = [
        o3d.geometry.TriangleMesh.create_sphere(
            0.005).translate(pos).paint_uniform_color(get_color(i, len(point))) for i, pos in enumerate(point)]
    poses_vis = []
    for p in pose:
        tcp_matrix = p.copy()
        print(tcp_matrix)
        tcp_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            0.03).transform(tcp_matrix)
        poses_vis.append(tcp_vis)
        from scripts.data_deform import L
        tcp_matrix[:3, 3] += tcp_matrix[:3, 2]*L
        tcp_vis = o3d.geometry.TriangleMesh.create_sphere(
            0.005).transform(tcp_matrix)
        poses_vis.append(tcp_vis)

    o3d.visualization.draw_geometries(
        [o, pcd, *points, *poses_vis, *other_vis])


def project(K, cam_to_d, points):
    # 先转换到相机坐标系
    d_to_cam = np.linalg.inv(cam_to_d)
    one = np.ones(list(points.shape[:-1])+[1])
    points = np.concatenate((points, one), -1)
    points = np.einsum('ij,bj->bi', d_to_cam, points)
    points = points[:, :3]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = points[:, 2]
    x = points[:, 0] * fx / z + cx
    y = points[:, 1] * fy / z + cy
    return np.stack((y, x, z), -1)


def vis2d(color, K, cam_to_d, points, name='a'):
    color = color.copy()
    if points is not None:
        xyz = project(K, cam_to_d, points)
        for i in range(len(points)):
            X = int(xyz[i, 0])
            Y = int(xyz[i, 1])
            # print(X, Y)
            color = cv2.circle(color, (Y, X), 5, get_color(
                i, len(points), to255=True), 2)
    imshow(name, color)
    # cv2.waitKey(1)
    return color


def viscdf(data_list, name=''):
    for i, data in enumerate(data_list):
        sorted_data = np.sort(data)
        cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, cumulative_prob, label=f'{i}')

    plt.xlabel('value')
    plt.ylabel('Cumulative Probability')
    plt.xlim(0, 2)
    # plt.ylim(0, 1)
    plt.legend()
    plt.yscale('log')
    plt.title(f'{name} CDF')
    plt.grid(True)
    plt.show()


def viscurve(data_list, name=''):
    for i, data in enumerate(data_list):
        # sorted_data = np.sort(data)
        # cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(data, label=f'{i}')

    plt.xlabel('t')
    plt.ylabel('v')
    # plt.xlim(0, 2)
    # plt.ylim(0, 1)
    plt.legend()
    # plt.yscale('log')
    plt.title(f'{name}')
    plt.grid(True)
    plt.show()


def vis_feature(points_list, method='pca'):
    points_list = points_list
    X = []
    Y = []
    for i, points in enumerate(points_list):
        X.extend(points)
        Y.extend([i] * len(points))
        # Y.extend([0 if i < 6 else (1 if i<10 else 2)] * len(points))
    # import ipdb;ipdb.set_trace()
    X = np.stack(X)[::100]
    # X = torch.stack(X).cpu().numpy()[::100]
    Y = np.array(Y)[::100]
    print(max(Y))
    C = get_colors(Y, max(Y))
    print(X.shape)

    if method == 'tsne' or method is None:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        X_tsne = tsne.fit_transform(X)

        plt.figure(figsize=(6, 5))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=C, s=5, alpha=0.6)
        plt.title('t-SNE Visualization')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.show()
    if method == 'pca' or method is None:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(6, 5))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=C, s=5, alpha=0.6)
        plt.title('PCA Visualization')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()

    if method == '3dpca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制散点，点小一些，带透明度
        sc = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
            c=C, cmap='tab10', s=3, alpha=0.6, edgecolors='none'
        )

        ax.set_title('PCA 3D Visualization')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        fig.colorbar(sc, label='Class')
        plt.show()
