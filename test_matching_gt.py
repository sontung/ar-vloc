import math
import sys

import PIL.Image
import cv2
import numpy as np

import colmap_io
import feature_matching
import point3d
import vis_utils
from scipy.spatial import KDTree

query_images_folder = "Test line"
sfm_images_dir = "sfm_ws_hblab/images.txt"
sfm_point_cloud_dir = "/home/sontung/work/recon_models/office/points3D.txt"
sfm_images_folder = "sfm_ws_hblab/images"
ground_truth_dir = "/home/sontung/work/recon_models/office/images.txt"
sfm_image_folder = "/home/sontung/work/recon_models/office/images"


def matching_gt():
    pid2tracks = colmap_io.read_points3D(sfm_point_cloud_dir)
    print(pid2tracks)

    image2pose_gt = colmap_io.read_images(ground_truth_dir)
    for im_id in image2pose_gt:
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose_gt[im_id]
        if "IMG_0761.HEIC" not in image_name:
            continue
        good_tracks = []
        img2oc = {}
        for pid in pid2tracks:
            tracks = pid2tracks[pid]
            new_track = []
            if im_id not in [tracks[du] for du in range(0, len(tracks), 2)]:
                continue
            for idx in range(0, len(tracks), 2):
                im_id2, fid = tracks[idx], tracks[idx+1]
                if "HEIC" not in image2pose_gt[im_id2][0] or im_id2 == im_id:
                    new_track.append((im_id2, fid))
            if len(new_track) > 1:
                good_tracks.append(new_track)
                print(new_track)
                for u, _ in new_track:
                    if u not in img2oc:
                        img2oc[u] = 1
                    else:
                        img2oc[u] += 1
        img_list = [60, 31]
        pairs = []
        for track in good_tracks:
            all_ids = [du[0] for du in track]
            if 31 in all_ids and 60 in all_ids:
                img2fid = {60: [], 31: []}
                for u, v in track:
                    if u in img_list:
                        img2fid[u].append(v)
                for fid1 in img2fid[60]:
                    for fid2 in img2fid[31]:
                        pairs.append([fid1, fid2])
        name1 = image2pose_gt[60][0]
        image1 = cv2.imread(f"{sfm_image_folder}/{name1}")
        name2 = image2pose_gt[31][0]
        image2 = cv2.imread(f"{sfm_image_folder}/{name2}")
        image = np.hstack([image1, image2])
        print(f"reading {name1, name2}")

        for fid1, fid2 in pairs:
            x1, y1, _ = image2pose_gt[60][1][fid1]
            x1, y1 = map(int, (x1, y1))
            cv2.circle(image, (x1, y1), 20, (128, 128, 0), -1)

            x2, y2, _ = image2pose_gt[31][1][fid2]
            x2, y2 = map(int, (x2, y2))
            cv2.circle(image, (x2+image1.shape[1], y2), 20, (128, 128, 0), -1)
            cv2.line(image, (x1, y1), (x2+image1.shape[1], y2), (0, 0, 0), 5)

        image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
        cv2.imshow("", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        sys.exit()
        for index, track in enumerate(good_tracks):
            visualized_list = []

            for im_id2, fid in track:
                name = image2pose_gt[im_id2][0]
                image = cv2.imread(f"{sfm_image_folder}/{name}")
                assert image is not None, f" cant read {sfm_image_folder}/{name}"
                x2, y2, _ = image2pose_gt[im_id2][1][fid]
                x2, y2 = map(int, (x2, y2))
                cv2.circle(image, (x2, y2), 50, (128, 128, 0), 20)
                visualized_list.append(image)

            list2 = []
            for im in visualized_list:
                im2 = PIL.Image.fromarray(im)
                im2.thumbnail((500, 500))
                im2 = np.array(im2)
                list2.append(im2)
            list2 = np.hstack(list2)
            cv2.imwrite(f"debug/gt/im-{index}.jpg", list2)


def normalize(dis):
    kDistNorm = 1.0 / (512.0 * 512.0)
    return math.acos(min(1.0, dis*kDistNorm))


def dot_custom(v1, v2):
    res = 0
    for idx in range(v1.shape[0]):
        res += int(v1[idx])*int(v2[idx])
    return res


def matching_pred():
    pairs = []
    descriptors, coordinates = feature_matching.load_2d_queries_using_colmap_sift_by_names(["IMG_0761.HEIC.jpg", "IMG_0726.jpg"])
    descriptors[0] = descriptors[0].astype(np.float64)
    descriptors[1] = descriptors[1].astype(np.float64)

    dist_mat = np.einsum('ij,kj->ik', descriptors[0], descriptors[1])
    for idx in range(descriptors[0].shape[0]):
        # best_i2 = -1
        # best_dist = 0
        # second_best_dist = 0
        #
        # for idx2 in range(descriptors[1].shape[0]):
        #     dist = dist_mat[idx, idx2]
        #     if dist > best_dist:
        #         best_i2 = idx2
        #         second_best_dist = best_dist
        #         best_dist = dist
        #     elif dist > second_best_dist:
        #         second_best_dist = dist
        #
        # if best_i2 == -1:
        #     continue

        best_i2 = np.argmax(dist_mat[idx, :])
        best_dist = dist_mat[idx, best_i2]
        row2 = np.delete(dist_mat[idx, :], best_i2)
        second_best_dist = np.max(row2)

        best_dist_norm = normalize(best_dist)

        if best_dist_norm > 0.7:
            continue

        second_best_dist_norm = normalize(second_best_dist)

        if best_dist_norm >= 0.8*second_best_dist_norm:
            continue

        print(best_dist_norm, idx, best_i2)
        pairs.append([coordinates[0][idx], coordinates[1][best_i2]])

    name1 = "IMG_0761.HEIC.jpg"
    image1 = cv2.imread(f"{sfm_image_folder}/{name1}")
    name2 = "IMG_0726.jpg"
    image2 = cv2.imread(f"{sfm_image_folder}/{name2}")
    image = np.hstack([image1, image2])
    print(f"reading {name1, name2}")

    for fid1, fid2 in pairs:
        x1, y1 = map(int, fid1)
        cv2.circle(image, (x1, y1), 20, (128, 128, 0), -1)

        x2, y2 = map(int, fid2)
        cv2.circle(image, (x2 + image1.shape[1], y2), 20, (128, 128, 0), -1)
        cv2.line(image, (x1, y1), (x2 + image1.shape[1], y2), (0, 0, 0), 5)

    image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
    cv2.imshow("", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    matching_pred()