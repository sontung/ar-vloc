import sys

import numpy as np
import open3d as o3d
import time
import cv2
from feature_matching import build_descriptors_2d, load_2d_queries_generic, load_2d_queries_using_colmap_sift
from feature_matching import build_descriptors_2d_using_colmap_sift, filter_using_d2_detector
from colmap_io import read_points3D_coordinates, read_images, read_cameras
from vis_utils import produce_cam_mesh
from point3d import PointCloud
from point2d import FeatureCloud
from vocab_tree import VocabTree
from localization import localize_single_image, localize_single_image_lt_pnp, localization_dummy
from optimizer import exhaustive_search

database = [(3242, 9124), (4004, 9173), (4021, 9035), (4288, 7432), (4523, 9173), (4533, 9124)]

VISUALIZING_SFM_POSES = False
VISUALIZING_POSES = True


query_images_folder = "Test line small"
sfm_images_dir = "sfm_ws_hblab/images.txt"
sfm_point_cloud_dir = "sfm_ws_hblab/points3D.txt"
sfm_images_folder = "sfm_ws_hblab/images"

image2pose = read_images(sfm_images_dir)
point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)
points_3d_list = []
point3d_id_list, point3d_desc_list, p3d_desc_list_multiple, point3did2descs = build_descriptors_2d_using_colmap_sift(image2pose)
point3d_cloud = PointCloud(point3did2descs)
for i in range(len(point3d_id_list)):
    point3d_id = point3d_id_list[i]
    point3d_desc = point3d_desc_list[i]
    xyzrgb = point3did2xyzrgb[point3d_id]
    point3d_cloud.add_point(point3d_id, point3d_desc, p3d_desc_list_multiple[i], xyzrgb[:3], xyzrgb[3:])
point3d_cloud.commit(image2pose)
point3d_cloud.cluster(image2pose)
point3d_cloud.build_desc_tree()
point3d_cloud.build_visibility_matrix(image2pose)
vocab_tree = VocabTree(point3d_cloud)
vocab_tree.load_matching_pairs(query_images_folder)

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

_, _, im_name_list, metadata_list, image_list, response_list = load_2d_queries_generic(query_images_folder)
desc_list, coord_list = load_2d_queries_using_colmap_sift(query_images_folder)

p2d2p3d = {}
for i in range(len(desc_list)):
    print(f"{im_name_list[i]}")
    start_time = time.time()
    point2d_cloud = FeatureCloud()
    for j in range(coord_list[i].shape[0]):
        point2d_cloud.add_point(j, desc_list[i][j], coord_list[i][j], 0.0)
    point2d_cloud.assign_words(vocab_tree.word2level, vocab_tree.v1)

    for pid, fid in database:
        point_cloud_vis = [point3d_cloud[pid].xyzrgb]
        pid_neighbors = point3d_cloud.xyz_nearest_and_covisible(pid, nb_neighbors=7)
        fid_neighbors = point2d_cloud.nearby_feature(fid, nb_neighbors=7)
        correct_pairs = [(pid_neighbors.index(pid), fid_neighbors.index(fid))]
        pid_desc_list = np.vstack([point3d_cloud[pid2].desc for pid2 in pid_neighbors])
        fid_desc_list = np.vstack([point2d_cloud[fid2].desc for fid2 in fid_neighbors])
        pid_coord_list = np.vstack([point3d_cloud[pid2].xyz for pid2 in pid_neighbors])
        fid_coord_list = np.vstack([point2d_cloud[fid2].xy for fid2 in fid_neighbors])
        sys.exit()

        # viz 3d neighbors
        for pid2 in pid_neighbors:
            point_cloud_vis.append(point3d_cloud[pid2].xyzrgb)
        point_cloud_vis = np.vstack(point_cloud_vis)
        point_cloud_vis[:, 3:] = [0, 0, 0]
        point_cloud_vis[0, 3:] = [1, 0, 0]
        point_cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud_vis[:, :3]))
        point_cloud_o3d.colors = o3d.utility.Vector3dVector(point_cloud_vis[:, 3:])
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1025)
        vis.add_geometry(point_cloud_o3d)
        vis.run()

        # viz 2d neighbors
        im_show = np.copy(image_list[0])
        fx, fy = map(int, point2d_cloud[fid].xy)
        cv2.circle(im_show, (fx, fy), 20, (128, 128, 0), 5)
        for fid2 in fid_neighbors:
            fx, fy = map(int, point2d_cloud[fid2].xy)
            cv2.circle(im_show, (fx, fy), 20, (0, 0, 0), 5)
        im_show = cv2.resize(im_show, (im_show.shape[1] // 4, im_show.shape[0] // 4))
        cv2.imshow("t", im_show)
        cv2.waitKey()
        cv2.destroyAllWindows()
        vis.destroy_window()
        break
