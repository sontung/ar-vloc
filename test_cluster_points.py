import sys
import numpy as np
import cv2
import random
import psutil
import time
import open3d as o3d
from feature_matching import build_descriptors_2d, load_2d_queries_opencv
from colmap_io import read_points3D_coordinates, read_images, read_cameras
from colmap_read import qvec2rotmat
from localization import localize_single_image
from scipy.spatial import KDTree
from vis_utils import produce_sphere, produce_cam_mesh, visualize_2d_3d_matching_single
from PIL import Image
from active_search import matching_active_search
from point3d import PointCloud, Point3D
from point2d import FeatureCloud
from vocab_tree import VocabTree
from cluster_algorithms import cluster_pos

MATCHING_BENCHMARK = True


def main():
    # query_images_folder = "sfm_ws_hblab/images"
    # query_images_folder = "/home/sontung/work/ar-vloc/Test line-20211207T083302Z-001/Test line"
    # cam_info_dir = "sfm_ws_hblab/cameras.txt"
    # sfm_images_dir = "sfm_ws_hblab/images.txt"
    # sfm_point_cloud_dir = "sfm_ws_hblab/points3D.txt"
    # sfm_images_folder = "sfm_ws_hblab/images"

    query_images_folder = "test_images"
    cam_info_dir = "sfm_models/cameras.txt"
    sfm_images_dir = "sfm_models/images.txt"
    sfm_point_cloud_dir = "sfm_models/points3D.txt"
    sfm_images_folder = "sfm_models/images"

    image2pose = read_images(sfm_images_dir)
    point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)
    points_3d_list = []
    point3d_id_list, point3d_desc_list = build_descriptors_2d(image2pose, sfm_images_folder)

    point3d_cloud = PointCloud(debug=MATCHING_BENCHMARK)
    for i in range(len(point3d_id_list)):
        point3d_id = point3d_id_list[i]
        point3d_desc = point3d_desc_list[i]
        xyzrgb = point3did2xyzrgb[point3d_id]
        point3d_cloud.add_point(point3d_id, point3d_desc, xyzrgb[:3], xyzrgb[3:])
    point3d_cloud.commit()
    cluster_pos(point3d_cloud)

    word2color = {}
    for point in point3d_cloud:
        x, y, z = point.xyz
        word = point.visual_word
        if word not in word2color:
            word2color[word] = (random.random(), random.random(), random.random())
        r, g, b = word2color[word]
        points_3d_list.append([x, y, z, r, g, b])
    points_3d_list = np.vstack(points_3d_list)
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d_list[:, :3]))
    point_cloud.colors = o3d.utility.Vector3dVector(points_3d_list[:, 3:])
    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1025)

    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    cameras = [point_cloud, coord_mesh]

    for c in cameras:
        vis.add_geometry(c)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
