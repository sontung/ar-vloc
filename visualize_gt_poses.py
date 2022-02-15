import os
import pickle
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
         sfm_point_cloud_dir="/home/sontung/work/ar-vloc/colmap_loc/sparse/0/points3D.txt",
         db_dir="/home/sontung/work/ar-vloc/colmap_loc/database.db"):
    image2pose_gt = read_images(sfm_images_dir)
    name2pose_gt = {}
    for im_id in image2pose_gt:
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose_gt[im_id]
        name2pose_gt[image_name] = cam_pose
        print(image_name)

    image2pose = read_images(sfm_images_dir)
    point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)
    points_3d_list = []
    point3d_id_list, point3d_desc_list, p3d_desc_list_multiple, point3did2descs = build_descriptors_2d_using_colmap_sift(image2pose, db_path=db_dir)
    point3d_cloud = PointCloud(point3did2descs, 0, 0, 0)
    for i in range(len(point3d_id_list)):
        point3d_id = point3d_id_list[i]
        point3d_desc = point3d_desc_list[i]
        xyzrgb = point3did2xyzrgb[point3d_id]
        point3d_cloud.add_point(point3d_id, point3d_desc, p3d_desc_list_multiple[i], xyzrgb[:3], xyzrgb[3:])
    point3d_cloud.commit(image2pose)
    point3d_cloud.cluster(image2pose)
    point3d_cloud.build_desc_tree()
    point3d_cloud.build_visibility_matrix(image2pose)


    if VISUALIZING_POSES:
        for point3d_id in point3did2xyzrgb:
            x, y, z, r, g, b = point3did2xyzrgb[point3d_id]
            if point3d_id in point3d_id_list:
                point3did2xyzrgb[point3d_id] = [x, y, z, 255, 0, 0]
            else:
                point3did2xyzrgb[point3d_id] = [x, y, z, 0, 0, 0]

        for point3d_id in point3did2xyzrgb:
            x, y, z, r, g, b = point3did2xyzrgb[point3d_id]
            points_3d_list.append([x, y, z, r/255, g/255, b/255])
        points_3d_list = np.vstack(points_3d_list)
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d_list[:, :3]))
        point_cloud.colors = o3d.utility.Vector3dVector(points_3d_list[:, 3:])
        point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

    im_names = ["query.jpg"]

    p2d2p3d = {}
    for i in range(len(im_names)):
        p2d2p3d[i] = [[], [], []]

        # gt
        p2d2p3d[i][2].append(name2pose_gt[f"{im_names[i]}"])

    localization_results = []
    for im_idx in p2d2p3d:
        qw, qx, qy, qz, tx, ty, tz = p2d2p3d[im_idx][2][0]
        ref_rot_mat = colmap_read.qvec2rotmat([qw, qx, qy, qz])
        t_vec = np.array([tx, ty, tz])
        t_vec = t_vec.reshape((-1, 1))
        localization_results.append(((ref_rot_mat, t_vec), (0, 0, 0)))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1025)

    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    cameras = [point_cloud, coord_mesh]

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
    main("/home/sontung/work/ar-vloc/colmap_loc/new/images.txt",
         "/home/sontung/work/ar-vloc/colmap_loc/new/points3D.txt",
         "/home/sontung/work/ar-vloc/colmap_loc/database.db")