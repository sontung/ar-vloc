import sys
import time

import cv2
import gc
import numpy as np
import kornia
import torch
import sqlite3
from PIL import Image


def read_points3D(in_dir="sfm_models/points3D.txt"):
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
        data[point3d_id] = [tracks]
    return data


def read_images(in_dir="sfm_models/images.txt"):
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

            data[image_id] = [image_name, points2d_meaningful, cam_pose, cam_id]
            print(f"loading {len(points2d_meaningful)} keypoints of image {image_id}")

            idx += 2
    return data


def decode_descriptors(db_dir="sfm_models/database.db"):
    conn = sqlite3.connect(db_dir)
    c = conn.cursor()
    data = list(c.execute("SELECT * from descriptors;"))
    conn.close()
    decoded_data = {}
    for im in data:
        print(f"loading {im[1]} descriptors of image {im[0]}")
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
    patch_size = 32
    batch_size = 64
    sift_model = kornia.feature.SIFTDescriptor(patch_size, 8, 4)
    hardnet_model = kornia.feature.HardNet8()
    hardnet_model = hardnet_model.double()
    hardnet_model.eval()
    images = read_images()
    point3did2descs = {}
    start_time = time.time()
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
    print(f"built descriptor database for 3D points in {time.time()-start_time}")

    # testing
    for point3d_id in point3did2descs:
        data = point3did2descs[point3d_id]
        mean_vec = []
        for _, sift, hardnet in data:
            mean_vec.append(sift.numpy())
        for _, sift, hardnet in data:
            mean_vec.append(sift.numpy())
            print(np.sum(np.abs(sift.numpy()-mean_vec)))


def visualize_matching_pairs():
    point3d = read_points3D()
    images = read_images()
    images_mat = [cv2.imread(f"sfm_models/images/{images[u][0]}") for u in images]
    track_lengths = []
    descriptors = decode_descriptors()
    for point3d_id in point3d:
        tracks = point3d[point3d_id][0]
        images_to_visualized = []
        # if len(tracks) <= 10:
        #     continue
        all_desc = []
        for i in range(0, len(tracks), 2):
            image_id = tracks[i]
            point2d_id = tracks[i+1]
            desc_mat = descriptors[image_id][-1]
            desc = desc_mat[point2d_id, :]
            all_desc.append(desc)
            image = images_mat[image_id-1].copy()
            # image = cv2.imread(f"sfm_models/images/{images[image_id][0]}")
            px, py, point3d_id2 = images[image_id][1][point2d_id]
            cv2.circle(image, (int(px), int(py)), 20, (255, 0, 0), -1)

            images_to_visualized.append(image)
        desc_mean = np.mean(all_desc, axis=0)
        for desc in all_desc:
            print(desc)
            print(np.sum(np.abs(desc-desc_mean)))

        # final_image = np.hstack(images_to_visualized)
        # final_image = cv2.resize(final_image, (final_image.shape[1]//4,
        #                                        final_image.shape[0]//4))
        # cv2.imshow("t", final_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # sys.exit()


if __name__ == '__main__':
    build_descriptors()
    # build_sfm_database()
    # visualize_matching_pairs()
