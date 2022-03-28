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


def read_gt_poses(pose_dir):
    sys.stdin = open(pose_dir, "r")
    lines = sys.stdin.readlines()
    poses = []
    for line in lines:
        line = line[:-1]
        name, qw, qx, qy, qz, tx, ty, tz = line.split(" ")
        cam_pose = list(map(float, [qw, qx, qy, qz, tx, ty, tz]))
        poses.append(cam_pose)
    return poses


def main(poses,
         sfm_images_dir="/home/sontung/work/ar-vloc/colmap_loc/sparse/0/images.txt",
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

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1025)

    # queried poses
    cameras = []
    for pose in poses:
        qw, qx, qy, qz, tx, ty, tz = pose
        # qx, qy, qz, qw, tx, ty, tz = pose

        rot_mat = colmap_read.qvec2rotmat([qw, qx, qy, qz])
        t_vec = np.array([tx, ty, tz])
        t_vec = t_vec.reshape((-1, 1))
        mat = np.hstack([-rot_mat.transpose(), t_vec])
        mat = np.vstack([mat, np.array([0, 0, 0, 1])])

        cm = produce_cam_mesh(color=(1, 0, 0))

        vertices = np.asarray(cm.vertices)
        for i in range(vertices.shape[0]):
            arr = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1])
            arr = mat @ arr
            vertices[i] = arr[:3]
        cm.vertices = o3d.utility.Vector3dVector(vertices)
        cameras.append(cm)

    for c in cameras[:5]:
        vis.add_geometry(c)

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    poses_gt = read_gt_poses("/home/sontung/work/Hierarchical-Localization/outputs/hblab/"
                             "Aachen_hloc_superpoint+superglue_netvlad20.txt")
    main(poses_gt,
         "/media/sontung/580ECE740ECE4B28/recon_models2/indoor2/dense/sparse/images.txt",
         "/media/sontung/580ECE740ECE4B28/recon_models2/indoor2/dense/sparse/points3D.txt")