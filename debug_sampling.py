import sys

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
image_ori = point2d_cloud.cluster(debug=True)
down_scale = 5
image_ori = cv2.resize(image_ori, (image_ori.shape[1]//down_scale, image_ori.shape[0]//down_scale))

image = image_ori.copy()
image = image[:, :, 0]*0.0

feature_indices = list(range(len(point2d_cloud)))
feature_probabilities = np.array([1/len(feature_indices) for _ in range(len(feature_indices))])
retired_list = set([])
x_coords = [int(point2d_cloud[idx].xy[1]//down_scale)-1 for idx in feature_indices]
y_coords = [int(point2d_cloud[idx].xy[0]//down_scale)-1 for idx in feature_indices]

# for _ in range(5):
while True:
    fid = np.random.choice(feature_indices, p=feature_probabilities / np.sum(feature_probabilities))
    if fid in retired_list:
        continue
    else:
        retired_list.add(fid)
    data1 = point3d_cloud.matching_2d_to_3d_brute_force_no_ratio_test(point2d_cloud[fid].desc)
    point_ind, dist, _, ratio = data1
    nb_desc = len(point3d_cloud[point_ind].multi_desc_list)
    if nb_desc > 2:
        add = nb_desc*0.01
    else:
        add = 0
    feature_probabilities[fid] += add

    print(f"{fid}, desc={nb_desc}")
    neighbors = point2d_cloud.nearby_feature(fid, nb_neighbors=nb_desc*10,
                                             min_distance=0, max_distance=nb_desc*10,
                                             strict_lower_bound=True)
    for new_feature_ind in neighbors:
        if new_feature_ind not in retired_list:
            feature_probabilities[new_feature_ind] += add/2

    feature_probabilities2 = feature_probabilities/np.max(feature_probabilities)

    image[x_coords, y_coords] = feature_probabilities2*255.0
    x, y = list(map(int, point2d_cloud[fid].xy))
    cv2.circle(image_ori, (x // down_scale, y // down_scale), 5, (0, 255, 0), -1)
    cv2.imshow("t", image)
    cv2.imshow("t2", image_ori)
    cv2.waitKey(1)

