import random
import sys
import time
import json
import cv2
import numpy as np
import kornia
import torch
import sqlite3
import struct
from PIL import Image

WARNING = False


def read_points3D(in_dir="sfm_models/points3D.txt"):
    """
    pid => pid, xyz, rgb
    """
    sys.stdin = open(in_dir, "r")
    lines = sys.stdin.readlines()
    data = {}
    for line in lines:
        if line[0] == "#":
            continue
        numbers = line[:-1].split(" ")
        numbers = list(map(float, numbers))
        point3d_id, x, y, z, r, g, b = numbers[:7]
        tracks = list(map(int, numbers[8:]))
        point3d_id = int(point3d_id)
        data[point3d_id] = tracks
    return data


def read_image_list(in_dir):
    sys.stdin = open(in_dir, "r")
    lines = sys.stdin.readlines()
    data = []
    for line in lines:
        data.append(line[:-1])
    return data


def read_points3D_coordinates(in_dir="sfm_models/points3D.txt", return_mat=False):
    """
    mapper from pid to xyz rgb
    """
    sys.stdin = open(in_dir, "r")
    lines = sys.stdin.readlines()
    data = {}
    for line in lines:
        if line[0] == "#":
            continue
        numbers = line[:-1].split(" ")[:7]
        numbers = list(map(float, numbers))
        point3d_id, x, y, z, r, g, b = numbers[:7]
        point3d_id = int(point3d_id)
        data[point3d_id] = [x, y, z, r, g, b]
    if return_mat:
        coord_mat = np.zeros((len(data), 3))
        color_mat = np.zeros((len(data), 3))
        id_mat = np.zeros((len(data),))
        for idx, pid in enumerate(data.keys()):
            x, y, z, r, g, b = data[pid]
            id_mat[idx] = pid
            coord_mat[idx] = [x, y, z]
            color_mat[idx] = [r, g, b]
        return data, coord_mat, color_mat, id_mat
    return data


def build_co_visibility_graph(image2pose):
    """
    co-visibility matrix between database images
    image_id_to_visibilities: image_id -> image_id2 -> number of co-visible points
    image_id_to_top_k: image_id -> list of co-visible images (sorted by the number of co-visible points)
    """
    pid2image_id = {}
    image_id_to_visibilities = {}
    for image_id in image2pose:
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose[image_id]
        image_id_to_visibilities[image_id] = {}
        for x, y, p3d_id in points2d_meaningful:
            if p3d_id > 0:
                if p3d_id not in pid2image_id:
                    pid2image_id[p3d_id] = [image_id]
                else:
                    pid2image_id[p3d_id].append(image_id)

    for pid in pid2image_id:
        images = pid2image_id[pid]
        for image_id in images:
            for image_id2 in images:
                if image_id2 != image_id:
                    if image_id2 not in image_id_to_visibilities[image_id]:
                        image_id_to_visibilities[image_id][image_id2] = 1
                    else:
                        image_id_to_visibilities[image_id][image_id2] += 1

    image_id_to_top_k = {}
    for image_id in image_id_to_visibilities:
        visibilities = image_id_to_visibilities[image_id]
        images = list(visibilities.keys())
        images = sorted(images, key=lambda du: visibilities[du], reverse=True)
        image_id_to_top_k[image_id] = images
    return image_id_to_visibilities, image_id_to_top_k


def read_cameras(cam_dir="sfm_ws_hblab/cameras.txt"):
    sys.stdin = open(cam_dir, "r")
    lines = sys.stdin.readlines()
    data = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line[0] == "#":
            idx += 1
            continue
        else:
            line = line[:-1].split(" ")
            cam_id, model, width, height = line[:4]
            cam_id, width, height = map(int, [cam_id, width, height])
            params = list(map(float, line[4:]))
            data[cam_id] = [model, width, height, params]
            idx += 1
    return data


def read_name2id(image2pose):
    name2id = {}
    for img_id in image2pose:
        img_name = image2pose[img_id][0]
        name2id[img_name] = img_id
    return name2id


def read_images(in_dir="sfm_models/images.txt", by_im_name=False):
    """
    this returns a dict:
    data[image_id] = [image_name, points2d_meaningful, cam_pose, cam_id]
    """
    sys.stdin = open(in_dir, "r")
    lines = sys.stdin.readlines()
    data = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line[0] == "#":
            idx += 1
            continue
        else:
            image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, image_name = line[:-1].split(" ")
            cam_pose = list(map(float, [qw, qx, qy, qz, tx, ty, tz]))
            image_id, cam_id = list(map(int, [image_id, cam_id]))
            points2d = list(map(float, lines[idx+1][:-1].split(" ")))
            points2d_meaningful = []  # [x, y, point 3d id]
            for i in range(0, len(points2d), 3):
                point = (points2d[i], points2d[i+1], int(points2d[i+2]))
                points2d_meaningful.append(point)
            if not by_im_name:
                data[image_id] = [image_name, points2d_meaningful, cam_pose, cam_id]
            else:
                data[image_name] = cam_pose
            idx += 2
    return data


def read_pid2images(image2pose):
    """
    maps point 3d id to 2d features that see this point.
    """
    data = {}
    for image_id in image2pose:
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose[image_id]
        for x, y, p3d_id in points2d_meaningful:
            if p3d_id > 0:
                if p3d_id not in data:
                    data[p3d_id] = [(image_id, image_name, x, y)]
                else:
                    data[p3d_id].append((image_id, image_name, x, y))
    return data


def decode_descriptors(db_dir="sfm_models/database.db"):
    conn = sqlite3.connect(db_dir)
    c = conn.cursor()
    data = list(c.execute("SELECT * from descriptors;"))
    conn.close()
    decoded_data = {}
    for im in data:
        print(f"Loading {im[1]} descriptors of image {im[0]}")
        res = []
        for u in im:
            if type(u) == bytes:
                res.append(np.frombuffer(u, dtype=np.uint8).reshape(-1, 128).astype(float))
            else:
                res.append(u)
        decoded_data[im[0]] = res
    return decoded_data


def decode_keypoints(data):
    decoded_data = []
    for im in data:
        res = []
        for u in im:
            if type(u) == bytes:
                res.append(np.frombuffer(u, dtype=np.float32))
            else:
                res.append(u)
        decoded_data.append(res)
    return decoded_data


def build_sfm_database():
    descriptors = decode_descriptors()
    point3d = read_points3D()
    images = read_images()
    for point3d_id in point3d:
        tracks = point3d[point3d_id][0]
        all_desc = []
        for i in range(0, len(tracks), 2):
            image_id = tracks[i]
            point2d_id = tracks[i + 1]
            desc_mat = descriptors[image_id][-1]
            desc = desc_mat[point2d_id, :]
            all_desc.append(desc)
            assert len(images[image_id][1]) == desc_mat.shape[0]
    return


def build_descriptors():
    patch_size = 41
    sift_model = kornia.feature.SIFTDescriptor(patch_size, 8, 4)
    images = read_images()
    point3did2descs = {}
    start_time = time.time()
    imageid2point3did2feature = {image_id: {} for image_id in images}
    for image_id in images:
        patches2d = []
        point3d_id_list = []
        image_name = images[image_id][0]

        an_img_gray = cv2.imread(f"sfm_models/images/{image_name}", cv2.IMREAD_GRAYSCALE)
        an_img_gray = np.array(an_img_gray, np.float64)/255.0
        an_img_gray = np.pad(an_img_gray, pad_width=patch_size)
        for y, x, point3d_id in images[image_id][1]:
            if point3d_id > 0:
                x, y = map(int, (x+patch_size, y+patch_size))
                patch = an_img_gray[x-patch_size//2: x+patch_size//2, y-patch_size//2: y+patch_size//2]
                try:
                    assert patch.shape == (patch_size, patch_size)
                except AssertionError:
                    cv2.imshow("t", patch)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                    raise AssertionError
                patches2d.append(np.expand_dims(patch, -1))
                point3d_id_list.append(point3d_id)
        patches2d = kornia.utils.image_list_to_tensor(patches2d)
        with torch.no_grad():
            sift_descs = sift_model.forward(patches2d)

        for i, point3d_id in enumerate(point3d_id_list):
            if point3d_id not in point3did2descs:
                point3did2descs[point3d_id] = [[image_id, sift_descs[i]]]
            else:
                point3did2descs[point3d_id].append([image_id, sift_descs[i]])

            if point3d_id in imageid2point3did2feature[image_id] and WARNING:
                print(f"Point {point3d_id} not unique in image {image_id}")
            imageid2point3did2feature[image_id][point3d_id] = [sift_descs[i]]
    print(f"Built descriptor database for 3D points in {time.time()-start_time}")

    # testing
    feature_to_use = 1
    accuracy = []
    nb_outliers = 20
    for point3d_id in point3did2descs:
        data = point3did2descs[point3d_id]
        image_id_list = [du[0] for du in data[1:]]
        f1 = data[0][feature_to_use]
        for i in image_id_list:
            f2 = imageid2point3did2feature[i][point3d_id][feature_to_use-1]
            diff1 = torch.sum(torch.square(f1-f2))
            count = 0.0
            samples = 0.0
            for _ in range(nb_outliers):
                other_point3d_id = random.choice(list(imageid2point3did2feature[i].keys()))
                if other_point3d_id != point3d_id:
                    other_feature_vec = imageid2point3did2feature[i][other_point3d_id][feature_to_use-1]
                    diff2 = torch.sum(torch.square(f1-other_feature_vec))
                    samples += 1
                    if diff2 > diff1:
                        count += 1
            accuracy.append(count/samples)
    print(f"Database descriptors accuracy is {np.mean(accuracy)} for {nb_outliers} outliers.")

    # return
    # imageid2point3did2feature: [image id] : [point3d id] : [all descriptors]
    # point3did2descs: [point3d id] : [all descriptors for all images] = (image_id, sift_descs, hardnet_descs)
    return imageid2point3did2feature, point3did2descs


def build_descriptors2():
    patch_size = 32
    batch_size = 64
    sift_model = kornia.feature.SIFTDescriptor(patch_size, 8, 4)
    hardnet_model = kornia.feature.HardNet8()
    hardnet_model = hardnet_model.double()
    hardnet_model.eval()
    images = read_images()
    point3did2descs = {}
    start_time = time.time()
    imageid2point3did2feature = {image_id: {} for image_id in images}
    for image_id in images:
        patches2d = []
        point3d_id_list = []
        image_name = images[image_id][0]

        an_img_gray = cv2.imread(f"sfm_models/images/{image_name}", cv2.IMREAD_GRAYSCALE)
        an_img_gray = np.array(an_img_gray, np.float64)/255.0
        an_img_gray = np.pad(an_img_gray, pad_width=patch_size)
        for y, x, point3d_id in images[image_id][1]:
            if point3d_id > 0:
                x, y = map(int, (x+patch_size, y+patch_size))
                patch = an_img_gray[x-patch_size//2: x+patch_size//2, y-patch_size//2: y+patch_size//2]
                try:
                    assert patch.shape == (patch_size, patch_size)
                except AssertionError:
                    cv2.imshow("t", patch)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                    raise AssertionError
                patches2d.append(np.expand_dims(patch, -1))
                point3d_id_list.append(point3d_id)
        patches2d = kornia.utils.image_list_to_tensor(patches2d)
        hardnet_descs = torch.zeros((patches2d.size(0), 128)).double()
        with torch.no_grad():
            sift_descs = sift_model.forward(patches2d)
            for i in range(0, patches2d.size(0), batch_size):
                start = i
                end = start + batch_size
                if end > patches2d.size(0):
                    end = patches2d.size(0)
                batch = patches2d[start: end]
                hardnet_descs[start: end] = hardnet_model.forward(batch).detach()

        for i, point3d_id in enumerate(point3d_id_list):
            if point3d_id not in point3did2descs:
                point3did2descs[point3d_id] = [[image_id, sift_descs[i], hardnet_descs[i]]]
            else:
                point3did2descs[point3d_id].append([image_id, sift_descs[i], hardnet_descs[i]])

            if point3d_id in imageid2point3did2feature[image_id] and WARNING:
                print(f"Point {point3d_id} not unique in image {image_id}")
            imageid2point3did2feature[image_id][point3d_id] = [sift_descs[i], hardnet_descs[i]]
    print(f"Built descriptor database for 3D points in {time.time()-start_time}")

    # testing
    sift_feature = 1
    hardnet_feature = 2
    feature_to_use = 1
    accuracy = []
    nb_outliers = 20
    for point3d_id in point3did2descs:
        data = point3did2descs[point3d_id]
        image_id_list = [du[0] for du in data[1:]]
        f1 = data[0][feature_to_use]
        for i in image_id_list:
            f2 = imageid2point3did2feature[i][point3d_id][feature_to_use-1]
            diff1 = torch.sum(torch.square(f1-f2))
            count = 0.0
            samples = 0.0
            for _ in range(nb_outliers):
                other_point3d_id = random.choice(list(imageid2point3did2feature[i].keys()))
                if other_point3d_id != point3d_id:
                    other_feature_vec = imageid2point3did2feature[i][other_point3d_id][feature_to_use-1]
                    diff2 = torch.sum(torch.square(f1-other_feature_vec))
                    samples += 1
                    if diff2 > diff1:
                        count += 1
            accuracy.append(count/samples)
    print(f"Database descriptors accuracy is {np.mean(accuracy)} for {nb_outliers} outliers.")

    # return
    # imageid2point3did2feature: [image id] : [point3d id] : [all descriptors]
    # point3did2descs: [point3d id] : [all descriptors for all images] = (image_id, sift_descs, hardnet_descs)
    return imageid2point3did2feature, point3did2descs


def visualize_matching_pairs():
    point3d = read_points3D("/home/sontung/work/hblab_office_reconstruction/points3D.txt")
    images = read_images("/home/sontung/work/hblab_office_reconstruction/images.txt")
    images_mat = {u: cv2.imread(f"/home/sontung/work/hblab_office_reconstruction/images/{images[u][0]}")
                  for u in images}
    track_lengths = []
    for point3d_id in point3d:
        tracks = point3d[point3d_id][0]
        images_to_visualized = []
        track_lengths.append(len(tracks))
        # if len(tracks) > 5:
        #     continue
        for i in range(0, len(tracks), 2):
            image_id = tracks[i]
            point2d_id = tracks[i+1]
            image = images_mat[image_id].copy()
            px, py, point3d_id2 = images[image_id][1][point2d_id]
            cv2.circle(image, (int(px), int(py)), 20, (255, 0, 0), -1)
            # image = cv2.resize(image, (1000, 500))
            images_to_visualized.append(image)

        final_image = np.hstack(images_to_visualized)
        final_image = cv2.resize(final_image, (final_image.shape[1]//4,
                                               final_image.shape[0]//4))
        # cv2.imshow("t", final_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # sys.exit()
    print(np.max(track_lengths), np.min(track_lengths))


def dump_image2pose_json():
    """
    this is for visualization by another independent application
    """
    image2pose = read_images()
    data = {}
    for image in image2pose:
        qw, qx, qy, qz, tx, ty, tz = image2pose[image][2]
        data[image] = {}
        data[image]["rotation"] = [qw, qx, qy, qz]
        data[image]["translation"] = [tx, ty, tz]
    data2 = {}
    point3did2xyzrgb = read_points3D_coordinates()
    for point3d_id in point3did2xyzrgb:
        x, y, z, r, g, b = point3did2xyzrgb[point3d_id]
        data2[point3d_id] = {}
        data2[point3d_id]["position"] = [x, y, z]
        data2[point3d_id]["color"] = [r, g, b]

    with open('data/points.json', 'w') as f:
        json.dump(data2, f)
    with open('data/image2pose.json', 'w') as f:
        json.dump(data, f)


def read_vocab_tree(a_file="/home/sontung/work/sfm_ws_hblab/vocab_tree_flickr100K_words32K.bin"):
    with open(a_file, mode='rb') as file:  # b is important -> binary
        fileContent = file.read()
    head = struct.unpack("QQf", fileContent[:20])
    print(head)
    body = struct.unpack("e"*20, fileContent[16:16+2*20])
    print(body)


if __name__ == '__main__':
    image2pose_ = read_images("/media/sontung/580ECE740ECE4B28/7scenes_reference_models/7scenes_reference_models/redkitchen/sfm_gt/images.txt", by_im_name=True)
    list_ = []
    for img in image2pose_:
        name = img.split("/")[0]
        if name not in list_:
            print(name)
            list_.append(name)
    # read_vocab_tree()
    # read_images("/home/sontung/work/sfm_ws_hblab/new_model/images.txt")
    # read_cameras()
    # dump_image2pose_json()
    # build_descriptors()
    # build_sfm_database()
    # visualize_matching_pairs()
