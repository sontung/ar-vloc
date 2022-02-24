import os
import pickle
import random
import sys
import pickle
import numpy as np
import open3d as o3d
import time
import cv2

import colmap_read
from feature_matching import build_descriptors_2d_using_colmap_sift, filter_using_d2_detector
from colmap_io import read_points3D_coordinates, read_images, read_cameras
from vis_utils import produce_cam_mesh
from point3d import PointCloud
from point2d import FeatureCloud
from vocab_tree import VocabTree
from localization import localize_single_image, localize_single_image_lt_pnp, localization_dummy


VISUALIZING_SFM_POSES = False
VISUALIZING_POSES = True


def main(sfm_images_dir="/home/sontung/work/ar-vloc/colmap_loc/sparse/0/images.txt",
         sfm_point_cloud_dir="/home/sontung/work/ar-vloc/colmap_loc/sparse/0/points3D.txt"):
    image2pose_gt = read_images(sfm_images_dir)
    name2pose_gt = {}
    for im_id in image2pose_gt:
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose_gt[im_id]
        name2pose_gt[image_name] = cam_pose

    image2pose = read_images(sfm_images_dir)
    point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)
    points_3d_list = []

    if VISUALIZING_POSES:
        for point3d_id in point3did2xyzrgb:
            x, y, z, r, g, b = point3did2xyzrgb[point3d_id]
            points_3d_list.append([x, y, z, r/255, g/255, b/255])
        points_3d_list = np.vstack(points_3d_list)
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d_list[:, :3]))
        point_cloud.colors = o3d.utility.Vector3dVector(points_3d_list[:, 3:])
        point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

    # im_names = ["query.jpg"]
    im_names = list(name2pose_gt.keys())

    p2d2p3d = {}
    idx = 480  # random.choice(range(0, len(im_names), 20))
    print(idx)
    p2d2p3d[idx] = [[], [], []]
    p2d2p3d[idx][2].append(name2pose_gt[f"{im_names[idx]}"])

    # for i in range(0, len(im_names), 20):
    #     p2d2p3d[i] = [[], [], []]
    #
    #     # gt
    #     p2d2p3d[i][2].append(name2pose_gt[f"{im_names[i]}"])

    localization_results = []
    for im_idx in p2d2p3d:
        qw, qx, qy, qz, tx, ty, tz = p2d2p3d[im_idx][2][0]
        print(qw, qx, qy, qz, tx, ty, tz)
        ref_rot_mat = colmap_read.qvec2rotmat([qw, qx, qy, qz])
        t_vec = np.array([tx, ty, tz])
        t_vec = t_vec.reshape((-1, 1))
        localization_results.append(((ref_rot_mat, t_vec), (0, 0, 0)))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1025)

    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    cameras = [point_cloud, coord_mesh]

    # mesh
    mesh = o3d.io.read_triangle_mesh("data/teapot.obj")
    mesh.scale(0.1, mesh.get_center())
    mesh.translate([0, 0, 0], relative=False)

    result, _ = localization_results[0]
    rot_mat, trans = result
    t = -rot_mat.transpose() @ trans
    t = t.reshape((3, 1))
    mat = np.hstack([-rot_mat.transpose(), t])
    mat = np.vstack([mat, np.array([0, 0, 0, 1])])
    vertices = np.asarray(mesh.vertices)
    for i in range(vertices.shape[0]):
        arr = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1])
        arr = mat @ arr
        vertices[i] = arr[:3]
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.translate([0, -1.2, -2.1], relative=True)

    vis.add_geometry(mesh)

    # queried poses
    for result, color_cam in localization_results:
        if result is None:
            continue
        rot_mat, trans = result
        t = -rot_mat.transpose() @ trans
        t = t.reshape((3, 1))
        mat = np.hstack([-rot_mat.transpose(), t])
        mat = np.vstack([mat, np.array([0, 0, 0, 1])])

        cm = produce_cam_mesh(color=color_cam)

        vertices = np.asarray(cm.vertices)
        for i in range(vertices.shape[0]):
            arr = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1])
            arr = mat @ arr
            vertices[i] = arr[:3]
        cm.vertices = o3d.utility.Vector3dVector(vertices)
        cameras.append(cm)

    for c in cameras:
        vis.add_geometry(c)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main("/home/sontung/work/recon_models/indoor/sparse/images.txt",
         "/home/sontung/work/recon_models/indoor/sparse/points3D.txt")