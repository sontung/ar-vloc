import os
import sys
import random
import torch
from PIL import Image
import numpy as np
import cv2
import time
import pydegensac
import sklearn.cluster
from scipy.spatial import KDTree
from sklearn.cluster import MiniBatchKMeans

MATCHING_BENCHMARK = True


def load_2d_queries_opencv(folder="test_images"):
    im_names = os.listdir(folder)
    descriptors = []
    coordinates = []
    for name in im_names:
        im_name = os.path.join(folder, name)
        im = cv2.imread(im_name)
        coord, desc = compute_kp_descriptors_opencv(im)
        coord = np.array(coord)
        coordinates.append(coord)
        descriptors.append(desc)
    return descriptors, coordinates, im_names


def build_descriptors_2d(images, images_folder="sfm_models/images"):
    point3did2descs = {}
    matching_ratio = []
    for image_id in images:
        image_name = images[image_id][0]
        image_name = f"{images_folder}/{image_name}"
        im = cv2.imread(image_name)
        coord, desc = compute_kp_descriptors_opencv(im)

        tree = KDTree(coord)
        total_dis = 0
        nb_points = 0
        nb_3d_points = 0
        points2d_meaningful = images[image_id][1]

        for x, y, p3d_id in points2d_meaningful:
            if p3d_id > 0:
                dis, idx = tree.query([x, y], 1)
                nb_3d_points += 1
                if dis < 2:
                    total_dis += dis
                    nb_points += 1
                    if p3d_id not in point3did2descs:
                        point3did2descs[p3d_id] = [[image_id, desc[idx]]]
                    else:
                        point3did2descs[p3d_id].append([image_id, desc[idx]])

        matching_ratio.append(nb_points/nb_3d_points)
    print(f"{round(np.mean(matching_ratio)*100, 3)}% of {len(point3did2descs)} 3D points found descriptors")
    p3d_id_list = []
    p3d_desc_list = []
    for p3d_id in point3did2descs:
        p3d_id_list.append(p3d_id)
        desc_list = [du[1] for du in point3did2descs[p3d_id]]
        desc = np.mean(desc_list, axis=0)
        p3d_desc_list.append(desc)
    return p3d_id_list, p3d_desc_list


def build_vocabulary_of_descriptors(p3d_id_list, p3d_desc_list, nb_clusters=None):
    if nb_clusters is None:
        nb_clusters = len(p3d_desc_list) // 50
    vocab = {u: [] for u in range(nb_clusters)}
    p3d_desc_list = np.array(p3d_desc_list)
    cluster_model = MiniBatchKMeans(nb_clusters)
    labels = cluster_model.fit_predict(p3d_desc_list)
    for i in range(len(p3d_id_list)):
        vocab[labels[i]].append((p3d_id_list[i], p3d_desc_list[i], i))
    return vocab, cluster_model


def compute_kp_descriptors_opencv(img, nb_keypoints=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if nb_keypoints is not None:
        sift = cv2.SIFT_create(edgeThreshold=10, contrastThreshold=0.02, nfeatures=nb_keypoints)
    else:
        sift = cv2.SIFT_create(edgeThreshold=10, contrastThreshold=0.02)
    kp_list, des = sift.detectAndCompute(img, None)
    coords = []
    for kp in kp_list:
        coords.append(list(kp.pt))
    return coords, des


def matching_2d_to_3d_brute_force(kd_tree, point3d_id_list, point3d_desc_list, query_desc):
    """
    brute forcing match for a single 2D point
    """
    res = kd_tree.query(query_desc, 2)
    if res[0][1] > 0.0:
        if res[0][0]/res[0][1] < 0.7:  # ratio test
            return point3d_id_list[res[1][0]]
    return None


def matching_2d_to_3d_vocab_based(point3d_id_list, point3d_desc_list,
                                  point2d_desc_list, cluster_model, vocab):
    """
    returns [image id] => point 2d id => point 3d id
    """
    start_time = time.time()
    result = {i: [] for i in range(len(point2d_desc_list))}
    matching_acc = []
    if MATCHING_BENCHMARK:
        kd_tree_3d = KDTree(point3d_desc_list)

    for i in range(len(point2d_desc_list)):
        desc_list = point2d_desc_list[i]

        # assign each desc to a word
        desc_list = np.array(desc_list)
        words = cluster_model.predict(desc_list)

        # sort feature by search cost
        features_to_match = [(du, desc_list[du], len(vocab[words[du]]), vocab[words[du]])
                             for du in range(desc_list.shape[0])]
        features_to_match = sorted(features_to_match, key=lambda du: du[2])

        count = 0
        for j, desc, _, point_3d_list in features_to_match:
            qu_point_3d_desc_list = [du2[1] for du2 in point_3d_list]
            qu_point_3d_id_list = [du2[0] for du2 in point_3d_list]

            kd_tree = KDTree(qu_point_3d_desc_list)
            res = kd_tree.query(desc, 2)
            if res[0][1] > 0.0:
                if res[0][0]/res[0][1] < 0.7:  # ratio test
                    result[i].append([j, qu_point_3d_id_list[res[1][0]]])
                    if MATCHING_BENCHMARK:
                        ref_res = matching_2d_to_3d_brute_force(kd_tree_3d, point3d_id_list, point3d_desc_list, desc)
                        if ref_res == qu_point_3d_id_list[res[1][0]]:
                            count += 1
            if len(result[i]) >= 100:
                matching_acc.append(count)
                break
    time_spent = time.time()-start_time
    if MATCHING_BENCHMARK:
        print(f"Matching accuracy={round(np.mean(matching_acc), 3)}%")
    print(f"Matching 2D-3D done in {round(time_spent, 3)} seconds, "
          f"avg. {round(time_spent/len(point2d_desc_list), 3)} seconds/image")
    return result


if __name__ == '__main__':
    build_descriptors_2d()