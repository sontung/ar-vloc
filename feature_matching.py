import os
import sys
import random
import torch
from tqdm import tqdm
from PIL import Image
import pyheif
import exifread
import numpy as np
import cv2
from scipy.spatial import KDTree
from sklearn.cluster import MiniBatchKMeans

MATCHING_BENCHMARK = True


def load_2d_queries_generic(folder):
    im_names = os.listdir(folder)
    descriptors = []
    coordinates = []
    md_list = []
    im_list = []
    for name in tqdm(im_names, desc="Reading query images"):
        metadata = {}
        im_name = os.path.join(folder, name)
        if "HEIC" in name:
            heif_file = pyheif.read(im_name)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            im = np.array(image)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            f = open(im_name, 'rb')
            tags = exifread.process_file(f)
            metadata["f"] = float(tags["EXIF FocalLengthIn35mmFilm"].values[0])
            metadata["cx"] = im.shape[1]/2
            metadata["cy"] = im.shape[0]/2

        else:
            im = cv2.imread(im_name)
        coord, desc = compute_kp_descriptors_opencv(im)
        coord = np.array(coord)
        coordinates.append(coord)
        descriptors.append(desc)
        md_list.append(metadata)
        im_list.append(im)
    return descriptors, coordinates, im_names, md_list, im_list


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
    for image_id in tqdm(images, desc="Loading descriptors of SfM model"):
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
    print(f"{round(np.mean(matching_ratio)*100, 3)}% of {len(point3did2descs)} 3D points found descriptors with "
          f"{round(np.mean([len(point3did2descs[du]) for du in point3did2descs]), 3)} descriptors/point")
    p3d_id_list = []
    p3d_desc_list = []
    for p3d_id in point3did2descs:
        p3d_id_list.append(p3d_id)
        desc_list = [du[1] for du in point3did2descs[p3d_id]]
        desc = np.mean(desc_list, axis=0)
        p3d_desc_list.append(desc)
    return p3d_id_list, p3d_desc_list, point3did2descs


def build_vocabulary_of_descriptors(p3d_id_list, p3d_desc_list, nb_clusters=None):
    if nb_clusters is None:
        nb_clusters = len(p3d_desc_list) // 50
    vocab = {u: [] for u in range(nb_clusters)}
    p3d_desc_list = np.array(p3d_desc_list)
    cluster_model = MiniBatchKMeans(nb_clusters, random_state=0)
    labels = cluster_model.fit_predict(p3d_desc_list)
    for i in range(len(p3d_id_list)):
        vocab[labels[i]].append((p3d_id_list[i], p3d_desc_list[i], i))
    return vocab, cluster_model


def compute_kp_descriptors_opencv(img, nb_keypoints=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if nb_keypoints is not None:
        sift = cv2.SIFT_create(edgeThreshold=10,
                               nOctaveLayers=4,
                               contrastThreshold=0.02,
                               nfeatures=nb_keypoints)
    else:
        sift = cv2.SIFT_create(edgeThreshold=10,
                               nOctaveLayers=4,
                               contrastThreshold=0.02)
    kp_list, des = sift.detectAndCompute(img, None)
    coords = []
    for kp in kp_list:
        coords.append(list(kp.pt))
    return coords, des


if __name__ == '__main__':
    load_2d_queries_generic("Test line")