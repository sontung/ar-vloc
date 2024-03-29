import os
import sys

import kmeans1d

sys.path.append("d2-net")
from extract_features import extract_using_d2_net
import pickle
from tqdm import tqdm
from PIL import Image
import pyheif
import exifread
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
from scipy.spatial import KDTree
from colmap_db_read import extract_colmap_sift

MATCHING_BENCHMARK = True


def load_2d_queries_minimal(folder):
    im_names = os.listdir(folder)
    md_list = []
    im_list = []
    name_list = []
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
            metadata["cx"] = im.shape[1] / 2
            metadata["cy"] = im.shape[0] / 2
        elif ".pkl" in name:
            continue
        else:
            im = cv2.imread(im_name)

        md_list.append(metadata)
        im_list.append(im)
        name_list.append(name)

    return name_list, md_list, im_list


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


def load_2d_queries_using_colmap_sift(folder, db_path="/home/sontung/work/ar-vloc/colmap_sift/test_small.db"):
    im_names = os.listdir(folder)
    descriptors = []
    coordinates = []
    id2kp, id2desc, id2name = extract_colmap_sift(db_path)
    name2id = {v: k for k, v in id2name.items()}
    for name in tqdm(im_names, desc="Reading query images"):
        if "jpg" not in name:
            continue
        db_id = name2id[name]
        kp = id2kp[db_id]
        desc = id2desc[db_id]
        desc = produce_root_sift(desc)
        coordinates.append(kp)
        descriptors.append(desc)
    return descriptors, coordinates


def load_2d_queries_using_colmap_sift_by_names(names, db_path="/home/sontung/work/ar-vloc/colmap_sift/test_small.db"):
    descriptors = []
    coordinates = []
    id2kp, id2desc, id2name = extract_colmap_sift(db_path)
    name2id = {v: k for k, v in id2name.items()}
    for name in tqdm(names, desc="Reading query images"):
        if "jpg" not in name:
            continue
        db_id = name2id[name]
        kp = id2kp[db_id]
        desc = id2desc[db_id]
        # desc = produce_root_sift(desc)
        coordinates.append(kp)
        descriptors.append(desc)
    return descriptors, coordinates


def build_descriptors_2d(images, images_folder="sfm_models/images"):
    point3did2descs = {}
    matching_ratio = []

    my_file = Path(f"{images_folder}/sfm_data.pkl")
    if my_file.is_file():
        with open(f"{images_folder}/sfm_data.pkl", 'rb') as handle:
            p3d_id_list, p3d_desc_list, p3d_desc_list_multiple, point3did2descs = pickle.load(handle)
    else:
        for image_id in tqdm(images, desc="Loading descriptors of SfM model"):
            image_name = images[image_id][0]
            image_name = f"{images_folder}/{image_name}"
            im = cv2.imread(image_name)
            coord, desc = compute_kp_descriptors_opencv(im)

            tree = KDTree(coord)
            nb_points = 0
            nb_3d_points = 0
            points2d_meaningful = images[image_id][1]

            for x, y, p3d_id in points2d_meaningful:
                if p3d_id > 0:
                    dis, idx = tree.query([x, y], 1)
                    nb_3d_points += 1
                    if dis < 2:
                        nb_points += 1
                        if p3d_id not in point3did2descs:
                            point3did2descs[p3d_id] = [[image_id, desc[idx]]]
                        else:
                            point3did2descs[p3d_id].append([image_id, desc[idx]])

            matching_ratio.append(nb_points / nb_3d_points)
        print(f"{round(np.mean(matching_ratio) * 100, 3)}% of {len(point3did2descs)} 3D points found descriptors with "
              f"{round(np.mean([len(point3did2descs[du]) for du in point3did2descs]), 3)} descriptors/point")
        p3d_id_list = []
        p3d_desc_list = []
        p3d_desc_list_multiple = []
        mean_diff = []
        for p3d_id in point3did2descs:
            p3d_id_list.append(p3d_id)
            desc_list = [du[1] for du in point3did2descs[p3d_id]]
            p3d_desc_list_multiple.append(desc_list)
            desc = np.mean(desc_list, axis=0)
            if len(desc_list) > 1:
                mean_diff.extend([np.sqrt(np.sum(np.square(desc - du))) for du in desc_list])
            p3d_desc_list.append(desc)
        print(f"Mean var. = {np.mean(mean_diff)}")
        with open(f"{images_folder}/sfm_data.pkl", 'wb') as handle:
            pickle.dump([p3d_id_list, p3d_desc_list, p3d_desc_list_multiple, point3did2descs],
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved SFM model to {images_folder}")
    return p3d_id_list, p3d_desc_list, p3d_desc_list_multiple, point3did2descs


def build_descriptors_2d_using_colmap_sift(images,
                                           db_path="/home/sontung/work/ar-vloc/sfm_ws_hblab/database.db"):
    point3did2descs = {}
    matching_ratio = []

    id2kp, id2desc, id2name = extract_colmap_sift(db_path)
    name2id = {v: k for k, v in id2name.items()}
    for image_id in tqdm(images, desc="Loading descriptors of SfM model"):
        image_name = images[image_id][0]
        db_id = name2id[image_name]
        coord, desc = id2kp[db_id], id2desc[db_id]
        desc = produce_root_sift(desc)
        assert coord.shape[0] == desc.shape[0]

        tree = KDTree(coord)
        nb_points = 0
        nb_3d_points = 0
        points2d_meaningful = images[image_id][1]

        for x, y, p3d_id in points2d_meaningful:
            if p3d_id > 0:
                dis, idx = tree.query([x, y], 1)
                nb_3d_points += 1
                if dis < 1:
                    nb_points += 1
                    if p3d_id not in point3did2descs:
                        point3did2descs[p3d_id] = [[image_id, desc[idx]]]
                    else:
                        point3did2descs[p3d_id].append([image_id, desc[idx]])

        matching_ratio.append(nb_points / nb_3d_points)
    print(f"{round(np.mean(matching_ratio) * 100, 3)}% of {len(point3did2descs)} 3D points found descriptors with "
          f"{round(np.mean([len(point3did2descs[du]) for du in point3did2descs]), 3)} descriptors/point")
    p3d_id_list = []
    p3d_desc_list = []
    p3d_desc_list_multiple = []
    mean_diff = []
    for p3d_id in point3did2descs:
        p3d_id_list.append(p3d_id)
        desc_list = [du[1] for du in point3did2descs[p3d_id]]
        p3d_desc_list_multiple.append(desc_list)
        desc = np.mean(desc_list, axis=0)
        if len(desc_list) > 1:
            mean_diff.extend([np.sqrt(np.sum(np.square(desc - du))) for du in desc_list])
        p3d_desc_list.append(desc)
    print(f"Mean var. = {np.mean(mean_diff)}")
    return p3d_id_list, p3d_desc_list, p3d_desc_list_multiple, point3did2descs


def build_descriptors_2d_using_colmap_sift_no_verbose(images,
                                                      db_path="/home/sontung/work/ar-vloc/sfm_ws_hblab/database.db"):
    point3did2descs = {}
    matching_ratio = []

    id2kp, id2desc, id2name = extract_colmap_sift(db_path)
    name2id = {v: k for k, v in id2name.items()}
    for image_id in images:
        image_name = images[image_id][0]
        db_id = name2id[image_name]
        coord, desc = id2kp[db_id], id2desc[db_id]
        desc = produce_root_sift(desc)
        assert coord.shape[0] == desc.shape[0]

        tree = KDTree(coord)
        nb_points = 0
        nb_3d_points = 0
        points2d_meaningful = images[image_id][1]

        for x, y, p3d_id in points2d_meaningful:
            if p3d_id > 0:
                dis, idx = tree.query([x, y], 1)
                nb_3d_points += 1
                if dis < 1:
                    nb_points += 1
                    if p3d_id not in point3did2descs:
                        point3did2descs[p3d_id] = [[image_id, desc[idx]]]
                    else:
                        point3did2descs[p3d_id].append([image_id, desc[idx]])

        matching_ratio.append(nb_points / nb_3d_points)
    p3d_id_list = []
    p3d_desc_list = []
    p3d_desc_list_multiple = []
    mean_diff = []
    for p3d_id in point3did2descs:
        p3d_id_list.append(p3d_id)
        desc_list = [du[1] for du in point3did2descs[p3d_id]]
        p3d_desc_list_multiple.append(desc_list)
        desc = np.mean(desc_list, axis=0)
        if len(desc_list) > 1:
            mean_diff.extend([np.sqrt(np.sum(np.square(desc - du))) for du in desc_list])
        p3d_desc_list.append(desc)
    return p3d_id_list, p3d_desc_list, p3d_desc_list_multiple, point3did2descs


def produce_root_sift(des):
    des = des.astype(np.float64)
    des /= (np.linalg.norm(des, axis=1, ord=2, keepdims=True) + 1e-7)
    des /= (np.linalg.norm(des, axis=1, ord=1, keepdims=True) + 1e-7)
    des = np.sqrt(des)
    return des


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


def compute_kp_descriptors_opencv(img, nb_keypoints=None, root_sift=True, debug=False, return_response=False):
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
    if debug:
        return kp_list, des, img
    des /= (np.linalg.norm(des, axis=1, ord=2, keepdims=True) + 1e-7)

    if root_sift:
        des /= (np.linalg.norm(des, axis=1, ord=1, keepdims=True) + 1e-7)
        des = np.sqrt(des)

    coords = []
    for kp in kp_list:
        coords.append(list(kp.pt))
    if return_response:
        response_list = [(kp.response, 1.0) for kp in kp_list]
        return coords, des, response_list
    return coords, des


def run_d2_detector_on_folder(images_folder, save_folder, image_list=None):
    precomputed_file = f"{save_folder}/d2_keypoints_db.pkl"
    my_file = Path(precomputed_file)
    if my_file.is_file():
        pass
    else:
        if image_list is None:
            image_list = os.listdir(images_folder)
        name2kp = {}
        for name in tqdm(image_list, desc="Running D2 detector on database images"):
            im_name = os.path.join(images_folder, name)
            img = cv2.imread(im_name)
            keypoints = d2_feature_detection(img)
            name2kp[name] = keypoints
        with open(precomputed_file, 'wb') as handle:
            pickle.dump(name2kp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return precomputed_file


def d2_feature_detection(img, vis=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    keypoints, responses = extract_using_d2_net(img)
    clusters, centroids = kmeans1d.cluster(responses, 2)
    clusters = np.array(clusters, dtype=bool)
    if centroids[0] < centroids[1]:
        select_ind = 1
    else:
        select_ind = 0
    strongest_keypoints = keypoints[clusters == select_ind, :2]
    if vis:
        for idx, (x2, y2) in enumerate(strongest_keypoints):
            x2, y2 = map(int, (x2, y2))
            cv2.circle(img, (x2, y2), 10, (128, 128, 0), -1)
        img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
        cv2.imshow("", img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return strongest_keypoints


if __name__ == '__main__':
    # img = cv2.imread("/home/sontung/work/ar-vloc/test_line_jpg/line-14.jpg")
    # d2_feature_detection(img, vis=True)
    run_d2_detector_on_folder("/home/sontung/work/ar-vloc/vloc_workspace_retrieval/images_retrieval/db",
                              "/home/sontung/work/ar-vloc/vloc_workspace_retrieval")
