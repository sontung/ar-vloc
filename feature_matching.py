import os
import sys
import random
import torch
from PIL import Image
import numpy as np
import kornia
import cv2
import time
import pydegensac
from scipy.spatial import KDTree


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


def compute_kp_descriptors_opencv(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(edgeThreshold=10, contrastThreshold=0.02)
    kp_list, des = sift.detectAndCompute(img, None)
    coords = []
    for kp in kp_list:
        coords.append(list(kp.pt))
    return coords, des


def matching_2d_to_3d_vocab_based(point3d_id_list, point3d_desc_list, point2d_desc_list, cluster_model, vocab):
    """
    returns [image id] => point 2d id => point 3d id
    """
    start_time = time.time()
    result = {i: [] for i in range(len(point2d_desc_list))}
    for i in range(len(point2d_desc_list)):
        desc_list = point2d_desc_list[i]

        # assign each desc to a word
        desc_list = np.array(desc_list)
        words = cluster_model.predict(desc_list)

        # sort feature by search cost
        features_to_match = [(du, desc_list[du], len(vocab[words[du]]), vocab[words[du]])
                             for du in range(desc_list.shape[0])]
        features_to_match = sorted(features_to_match, key=lambda du: du[2])
        for j, desc, _, point_3d_list in features_to_match:
            point_3d_desc_list = [du2[1] for du2 in point_3d_list]
            point_3d_id_list = [du2[0] for du2 in point_3d_list]

            kd_tree = KDTree(point_3d_desc_list)
            res = kd_tree.query(desc, 2)
            if res[0][1] > 0.0:
                if res[0][0]/res[0][1] < 0.7:  # ratio test
                    result[i].append([j, point_3d_id_list[res[1][0]]])
            if len(result[i]) >= 100:
                break
    time_spent = time.time()-start_time
    print(f"Matching 2D-3D done in {round(time_spent, 3)} seconds, "
          f"avg. {round(time_spent/len(point2d_desc_list), 3)} seconds/image")
    return result


if __name__ == '__main__':
    build_descriptors_2d()