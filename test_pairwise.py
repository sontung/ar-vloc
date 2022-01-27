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
from utils import angle_between
import matplotlib.pyplot as plt


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
point2d_cloud = FeatureCloud()
for j in range(coord_list[0].shape[0]):
    point2d_cloud.add_point(j, desc_list[0][j], coord_list[0][j], 0.0)
point2d_cloud.assign_words(vocab_tree.word2level, vocab_tree.v1)


def compute_pairwise_edge_cost(u1, v1, u2, v2, min_var_axis):
    u1 = np.array([u1[du] for du in [0, 1, 2] if du != min_var_axis])
    u2 = np.array([u2[du] for du in [0, 1, 2] if du != min_var_axis])

    vec1 = u1-v1
    vec2 = u2-v2
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cost2 = np.abs(norm2-norm1)/(norm2+norm1)
    cost1 = np.abs(angle_between(vec1, vec2))
    # print(cost1, cost2)
    return cost2


solutions_gt = [(3337, 455), (5004, 5865), (4241, 8781), (3602, 9089), (4374, 5848), (4000, 9170), (4001, 8166),
                (4004, 9173), (3242, 9124), (4907, 7674), (3243, 4024), (3244, 9121), (4142, 7693), (4141, 4174),
                (5168, 7702), (4144, 453), (4018, 9119), (4019, 5896), (4017, 8935), (4021, 5940), (4022, 9090),
                (4535, 9207), (3251, 4267), (3527, 9175), (3400, 4164), (4167, 4117), (3282, 8486), (9810, 4190),
                (3544, 649), (4056, 8760), (4057, 5914), (3546, 7009), (4058, 8756), (3303, 928), (3306, 4307),
                (4205, 874), (4206, 6044), (8178, 385)]

pid_list = [4004, 3337, 4907, 3243, 3244, 5168, 3282, 8178, 9810, 4535]
fid_list = [7687, 4107, 526, 7702, 7734, 7743, 577, 7745, 579, 4164, 7748, 4174, 592, 4179, 606, 4190, 4195, 4205,
            4211, 649, 657, 661, 4246, 667, 4251, 4261, 701, 713, 8928, 5857, 8935, 5865, 8940, 5871, 759, 760, 765,
            5894, 5914, 796, 6952, 811, 8494, 814, 819, 5944, 6970, 5950, 830, 8512, 8514, 6978, 8516, 834, 5966, 5969,
            6996, 7009, 5990, 3943, 874, 878, 9089, 385, 7043, 3972, 3990, 3991, 3992, 9118, 9121, 419, 9124, 4006,
            425, 4011, 4020, 4024, 4028, 9150, 4031, 455, 4053, 9173, 4062, 4070, 8168, 8181, 504, 505, 7674, 4092]


for pid, fid in solutions_gt:
    pid_coord = point3d_cloud[pid].xyz
    fid_coord_true = point2d_cloud[fid].xy
    fid_list2 = sorted(fid_list[:], key=lambda du: point2d_cloud[du].xy[0])
    tracks = []
    for count, fid2 in enumerate(fid_list2):
        fid_coord_false = point2d_cloud[fid2].xy
        cost = compute_pairwise_edge_cost(pid_coord, fid_coord_true, pid_coord, fid_coord_false, 2)

        pid_coord_wo_min_axis = np.array([pid_coord[du] for du in [0, 1, 2] if du != 2])
        x1, y1 = [pid_coord_wo_min_axis[0], fid_coord_true[0]], [pid_coord_wo_min_axis[1], fid_coord_true[1]]
        x2, y2 = [pid_coord_wo_min_axis[0], fid_coord_false[0]], [pid_coord_wo_min_axis[1], fid_coord_false[1]]
        tracks.append([cost, x1, y1, x2, y2])

    tracks = sorted(tracks, key=lambda du: du[0])
    for count, (cost, x1, y1, x2, y2) in enumerate(tracks):
        plt.plot(x1, y1, x2, y2, marker='o')
        plt.savefig(f"debug/pw/{count}-{cost}.png")
        plt.close()
    break
