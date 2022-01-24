import pickle
import sys
import pickle
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
from scipy.spatial import KDTree
from vis_utils import visualize_matching_helper, visualize_all_point_images


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY, down_scale, ACTIVATED
    ACTIVATED = False
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(image_ori, (x, y), 5, (255, 0, 0), 5)
        mouseX, mouseY = x*down_scale, y*down_scale
        ACTIVATED = True


ACTIVATED = False
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

    database = point3d_cloud.sample(point2d_cloud, image_list[i])

    ori_len = len(database)
    ori_database = database[:]
    pid_neighbors = []
    fid_neighbors = []
    correct_pairs = []
    for pid, fid, dis, ratio in ori_database:
        pid_neighbors.extend(point3d_cloud.xyz_nearest_and_covisible(pid, nb_neighbors=10))
        fid_neighbors.extend(point2d_cloud.nearby_feature(fid, nb_neighbors=100))
        correct_pairs.append((pid, fid))

    # filter duplicate
    pid_neighbors = list(set(pid_neighbors))
    fid_neighbors = list(set(fid_neighbors))
    must_fid = [pair[1] for pair in correct_pairs]
    fid_neighbors = point2d_cloud.filter_duplicate_features(fid_neighbors, must_fid)
    new_correct_pairs = correct_pairs[:]
    for index, (pid, fid) in enumerate(correct_pairs):
        new_correct_pairs[index] = (pid_neighbors.index(pid), fid_neighbors.index(fid))

    pid_desc_list = np.vstack([point3d_cloud[pid2].desc for pid2 in pid_neighbors])
    fid_desc_list = np.vstack([point2d_cloud[fid2].desc for fid2 in fid_neighbors])
    pid_coord_list = np.vstack([point3d_cloud[pid2].xyz for pid2 in pid_neighbors])
    fid_coord_list = np.vstack([point2d_cloud[fid2].xy for fid2 in fid_neighbors])
    fid_coord_list_ori = np.copy(fid_coord_list)
    fid_coord_tree = KDTree(fid_coord_list)
    fid_coord_tree_ori = KDTree(fid_coord_list_ori)

    image_ori = np.copy(image_list[i])
    down_scale = 5

    mouseX = -1
    mouseY = -1
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    for idx in range(fid_coord_list.shape[0]):
        fx, fy = fid_coord_list[idx]
        fx, fy = map(int, (fx, fy))
        cv2.circle(image_ori, (fx, fy), 10, (255, 0, 0), 3)
    image_ori = cv2.resize(image_ori, (image_ori.shape[1] // down_scale, image_ori.shape[0] // down_scale))

    result = []
    for index_pid in range(pid_coord_list.shape[0]):
        point_images = visualize_all_point_images(point3d_cloud[pid_neighbors[index_pid]], "sfm_ws_hblab/images")
        cv2.imshow("point", point_images)

        while True:

            cv2.imshow("image", image_ori)

            if mouseX > 0:
                _, idx = fid_coord_tree.query([mouseX, mouseY])
                fx, fy = fid_coord_list[idx] / down_scale
                fx, fy = map(int, (fx, fy))
                cv2.circle(image_ori, (fx, fy), 5, (128, 128, 0), -1)

            k = cv2.waitKey(20) & 0xFF

            if k == 27 or k == ord("q"):
                break
            elif k == ord('d'):
                if mouseX > 0:
                    _, index_fid = fid_coord_tree_ori.query(fid_coord_list[idx])

                    print(index_fid, index_pid, fid_coord_list[idx]-fid_coord_list_ori[index_fid])
                    fid_coord_list = np.delete(fid_coord_list, idx, 0)
                    fid_coord_tree = KDTree(fid_coord_list)
                    result.append((index_pid, index_fid))

                image_ori = np.copy(image_list[i])
                for idx in range(fid_coord_list.shape[0]):
                    fx, fy = fid_coord_list[idx]
                    fx, fy = map(int, (fx, fy))
                    cv2.circle(image_ori, (fx, fy), 10, (255, 0, 0), 3)
                image_ori = cv2.resize(image_ori, (image_ori.shape[1] // down_scale, image_ori.shape[0] // down_scale))
                mouseX = -1
                mouseY = -1
                MODE = None
                RESULT = {}
                break
        cv2.destroyWindow("point")
print(result)


