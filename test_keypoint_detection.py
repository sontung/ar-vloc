import sys
sys.path.append("d2-net")
from extract_features import extract_using_d2_net

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
from scipy.spatial import KDTree


query_images_folder = "Test line small"
sfm_images_dir = "sfm_ws_hblab/images.txt"
sfm_point_cloud_dir = "sfm_ws_hblab/points3D.txt"
sfm_images_folder = "sfm_ws_hblab/images"
image2pose = read_images(sfm_images_dir)
point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)


point3d_id_list, point3d_desc_list, p3d_desc_list_multiple, point3did2descs = build_descriptors_2d(image2pose, sfm_images_folder)

point3d_cloud = PointCloud(point3did2descs, debug=True)
for i in range(len(point3d_id_list)):
    point3d_id = point3d_id_list[i]
    point3d_desc = point3d_desc_list[i]
    xyzrgb = point3did2xyzrgb[point3d_id]
    point3d_cloud.add_point(point3d_id, point3d_desc, p3d_desc_list_multiple[i], xyzrgb[:3], xyzrgb[3:])

point3d_cloud.commit(image2pose)
vocab_tree = VocabTree(point3d_cloud)

descriptors, coordinates, im_names, md_list, im_list, response_list = load_2d_queries_generic(query_images_folder)
point2d_cloud = FeatureCloud(im_list[0])

for j in range(coordinates[0].shape[0]):
    point2d_cloud.add_point(j, descriptors[0][j], coordinates[0][j], response_list[0][j])
point2d_cloud.nearby_feature(0, 2)
point2d_cloud.rank_feature_strengths()

img = np.copy(im_list[0])

# draw d2 features
img2 = np.copy(im_list[0])
keypoints = extract_using_d2_net(img2)
for x, y, _ in keypoints:
    x, y = map(int, (x, y))
    cv2.circle(img2, (x, y), 5, (255, 0, 0), -1)

# draw sift features
tree = KDTree(keypoints[:, :2])
for j in range(coordinates[0].shape[0]):
    distance, idx = tree.query(coordinates[0][j], 1)
    if distance < 4:
        print(coordinates[0][j], keypoints[idx])
        x, y = map(int, coordinates[0][j])
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        x, y = map(int, keypoints[idx, :2])
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)


img2 = cv2.resize(img2, (img2.shape[1]//4, img2.shape[0]//4))
img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
img = np.hstack([img, img2])
cv2.imshow("t", img)
cv2.waitKey()
cv2.destroyAllWindows()
sys.exit()

NB_CLUSTERS = 5
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
cluster_probabilities_based_on_feature_strengths = np.ones((len(point2d_cloud.cid2kp),))*1/len(point2d_cloud.cid2kp)
for c in point2d_cloud.cid2prob:
    cluster_probabilities_based_on_feature_strengths[c] = point2d_cloud.cid2prob[c]
print(cluster_probabilities_based_on_feature_strengths)

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