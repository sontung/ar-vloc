import sys
import random
import matplotlib
matplotlib.use('TkAgg')
import cv2
import numpy as np
from feature_matching import load_2d_queries_generic
from point2d import FeatureCloud
from point3d import PointCloud
from colmap_io import read_points3D_coordinates, read_images
from feature_matching import build_descriptors_2d
from vocab_tree import VocabTree
from vis_utils import visualize_matching, visualize_matching_helper
from hardnet import build_descriptors_2d as build_descriptors_2d_hardnet
from hardnet import load_2d_queries_generic as load_2d_queries_generic_hardnet
from matplotlib import pyplot as plt
from matplotlib import animation


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY, mouseX2, mouseY2, down_scale
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(image, (x, y), 5, (255, 0, 0), 5)
        mouseX, mouseY = x*down_scale, y*down_scale


USING_HARDNET = False
query_images_folder = "Test line small"
sfm_images_dir = "sfm_ws_hblab/images.txt"
sfm_point_cloud_dir = "sfm_ws_hblab/points3D.txt"
sfm_images_folder = "sfm_ws_hblab/images"
image2pose = read_images(sfm_images_dir)
point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)

if USING_HARDNET:
    using_ps = True
    read_data = build_descriptors_2d_hardnet(image2pose,
                                             sfm_images_folder,
                                             using_ps=using_ps)
    point3d_id_list, point3d_desc_list, p3d_desc_list_multiple, point3did2descs = read_data
else:
    point3d_id_list, point3d_desc_list, p3d_desc_list_multiple, point3did2descs = build_descriptors_2d(image2pose, sfm_images_folder)

point3d_cloud = PointCloud(point3did2descs, debug=True)
for i in range(len(point3d_id_list)):
    point3d_id = point3d_id_list[i]
    point3d_desc = point3d_desc_list[i]
    xyzrgb = point3did2xyzrgb[point3d_id]
    point3d_cloud.add_point(point3d_id, point3d_desc, p3d_desc_list_multiple[i], xyzrgb[:3], xyzrgb[3:])

point3d_cloud.commit(image2pose)
vocab_tree = VocabTree(point3d_cloud)
if USING_HARDNET:
    read_queries = load_2d_queries_generic_hardnet(query_images_folder, using_ps=using_ps)
    descriptors, coordinates, im_names, md_list, im_list, response_list = read_queries
else:
    descriptors, coordinates, im_names, md_list, im_list, response_list = load_2d_queries_generic(query_images_folder)
point2d_cloud = FeatureCloud(im_list[0])

for j in range(coordinates[0].shape[0]):
    point2d_cloud.add_point(j, descriptors[0][j], coordinates[0][j], response_list[0][j])
point2d_cloud.nearby_feature(0, 2)
point2d_cloud.rank_feature_strengths()

NB_CLUSTERS = 100
image_ori = point2d_cloud.cluster(nb_clusters=NB_CLUSTERS, debug=True)
down_scale = 5
image_ori = cv2.resize(image_ori, (image_ori.shape[1]//down_scale, image_ori.shape[0]//down_scale))

image = image_ori.copy()
image = image[:, :, 0]*0.0

fid2cid = {}
for cid in point2d_cloud.cid2kp:
    fid_list, _ = point2d_cloud.cid2kp[cid]
    for fid in fid_list:
        fid2cid[fid] = cid
feature_indices = list(range(len(point2d_cloud)))
feature_strengths = np.array([1-abs(point2d_cloud[idx].strength_rank-0.5) for idx in feature_indices])
cluster_indices = list(range(len(point2d_cloud.cid2kp)))
cluster_probabilities = np.ones((len(point2d_cloud.cid2kp),))*1/len(point2d_cloud.cid2kp)
feature_probabilities = np.array([1/len(feature_indices) for _ in range(len(feature_indices))])
x_coords = [int(point2d_cloud[idx].xy[1]//down_scale)-1 for idx in feature_indices]
y_coords = [int(point2d_cloud[idx].xy[0]//down_scale)-1 for idx in feature_indices]
nb_desc_list = np.zeros_like(cluster_probabilities)
count_desc_list = np.zeros_like(cluster_probabilities)
r_list = np.zeros_like(feature_probabilities)
max_desc = 0
explore_prob = 0.5
recent_desc = []
record_cid = 0
record_list = []
desc_tracks = {}
desc_tracks2 = {}
cid_tracks = {cid: 0 for cid in cluster_indices}

# for _ in range(5):

# exploring
for _ in range(2):
    for cid in cluster_indices:
        fid_list, _ = point2d_cloud.cid2kp[cid]
        fid = np.random.choice(fid_list)
        if r_list[fid] == 1:
            continue
        r_list[fid] = 1
        data1 = point3d_cloud.matching_2d_to_3d_brute_force_no_ratio_test(point2d_cloud[fid].desc)
        point_ind, dist, _, ratio = data1
        nb_desc = len(point3d_cloud[point_ind].multi_desc_list)
        nb_desc_list[cid] += nb_desc
        count_desc_list[cid] += 1

# exploiting
for _ in range(1000):
    record = 0
    exploited = True
    cluster_probabilities = np.zeros((len(cluster_indices),))
    a1 = nb_desc_list / count_desc_list
    non_zero_idx = np.nonzero(a1 > 2)
    if non_zero_idx[0].shape[0] > 0:
        base_prob = 1 / len(cluster_indices)
        cluster_probabilities[non_zero_idx] = a1[non_zero_idx] * base_prob
    print(cluster_probabilities)
    prob_sum = np.sum(cluster_probabilities)
    if prob_sum <= 0.0:
        continue
    cid = np.random.choice(cluster_indices, p=cluster_probabilities/prob_sum)
    cid_tracks[cid] += 1
    fid_list, _ = point2d_cloud.cid2kp[cid]
    fid = np.random.choice(fid_list)

    if r_list[fid] == 1:
        continue
    r_list[fid] = 1
    data1 = point3d_cloud.matching_2d_to_3d_brute_force_no_ratio_test(point2d_cloud[fid].desc)
    point_ind, dist, _, ratio = data1
    nb_desc = len(point3d_cloud[point_ind].multi_desc_list)

    if nb_desc not in desc_tracks:
        desc_tracks[nb_desc] = 1
    else:
        desc_tracks[nb_desc] += 1

    cid = fid2cid[fid]
    if nb_desc > max_desc:
        record_cid = cid
        max_desc = nb_desc
        record = 1
        record_list.append(fid)
    nb_desc_list[cid] += nb_desc
    count_desc_list[cid] += 1

    recent_desc.insert(0, nb_desc)
    if len(recent_desc) > 1000:
        recent_desc.pop()
    print(f"desc={nb_desc} mean desc={np.round(np.mean(recent_desc), 3)} record={max_desc} "
          f"explore prob={np.round(explore_prob, 3)}")

    if exploited:
        if nb_desc not in desc_tracks2:
            desc_tracks2[nb_desc] = 1
        else:
            desc_tracks2[nb_desc] += 1
        x, y = list(map(int, point2d_cloud[fid].xy))
        cv2.circle(image_ori, (x // down_scale, y // down_scale), 5, (0, 0, 0), 2)
    for fid in record_list:
        x, y = list(map(int, point2d_cloud[fid].xy))
        cv2.circle(image_ori, (x // down_scale, y // down_scale), 5, (0, 0, 255), 2)
    cv2.imshow("t2", image_ori)
    cv2.waitKey(1)
print(desc_tracks)
print(desc_tracks2)
cid_list = sorted(cluster_indices, key=lambda du: cid_tracks[du])
data = []
for cid in cid_list:
    if cid_tracks[cid] > 0:
        print(cid_tracks[cid], nb_desc_list[cid]/count_desc_list[cid])
        data.append(nb_desc_list[cid]/count_desc_list[cid])

cv2.waitKey()
cv2.destroyAllWindows()
