import sys

import cv2
import numpy as np
import colmap_db_read
import colmap_io
from vis_utils import visualize_matching_pairs
from utils import to_homo


def proj_err(coord1, coord2, mat):
    coord1 = to_homo(coord1)
    coord2 = to_homo(coord2)
    v1 = mat @ coord1
    v2 = coord1 @ mat
    v1 /= v1[-1]
    v2 /= v2[-1]
    d1 = np.sum(np.square(v1-coord2))
    d2 = np.sum(np.square(v2-coord2))
    return d1, d2


def proj_err2(coord1, coord2, mat):
    coord1 = to_homo(coord1)
    coord2 = to_homo(coord2)
    v2 = coord1 @ mat
    return v2 @ coord2


class Localization:
    def __init__(self):
        self.match_database_dir = "colmap_sift/matching.db"  # dir to database containing matches between queries and database
        self.db_images_dir = "colmap_sift/db_images"
        return

    def read_2d_2d_matches(self, query_im_name):
        matches, geom_matrices = colmap_db_read.extract_colmap_two_view_geometries(self.match_database_dir)
        # matches = colmap_db_read.extract_colmap_matches(self.match_database_dir)
        id2kp, _, id2name = colmap_db_read.extract_colmap_sift(self.match_database_dir)
        name2id = {name: ind for ind, name in id2name.items()}
        query_im_id = name2id[query_im_name]
        for m in matches:
            if query_im_id in m:
                arr = matches[m]
                conf, (f_mat, e_mat, h_mat) = geom_matrices[m]

                if arr is not None and f_mat is not None:
                    pairs = []
                    id1, id2 = m
                    for u, v in arr:
                        coord1 = id2kp[id1][u]
                        coord2 = id2kp[id2][v]
                        coord1 = to_homo(coord1).astype(np.float64)
                        coord2 = to_homo(coord2).astype(np.float64)
                        dis = cv2.sampsonDistance(coord1, coord2,
                                                  f_mat.astype(np.float64))
                        if dis < 1:
                            pair = (list(map(int, id2kp[id1][u])), list(map(int, id2kp[id2][v])), dis)
                            pairs.append(pair)
                    image1 = cv2.imread(f"{self.db_images_dir}/{id2name[id1]}")
                    image2 = cv2.imread(f"{self.db_images_dir}/{id2name[id2]}")
                    image = visualize_matching_pairs(image1, image2, pairs)
                    cv2.imwrite(f"colmap_sift/matches/img-{id1}-{id2}.jpg", image)
                    # image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
                    # cv2.imshow("", image)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
        sys.exit()
        query_images_folder = "test_line_jpg"
        sfm_images_dir = "sfm_ws_hblab/images.txt"
        sfm_point_cloud_dir = "/home/sontung/work/recon_models/office/points3D.txt"
        sfm_images_folder = "sfm_ws_hblab/images"
        ground_truth_dir = "/home/sontung/work/recon_models/office/images.txt"
        sfm_image_folder = "/home/sontung/work/recon_models/office/images"

        pid2tracks = colmap_io.read_points3D(sfm_point_cloud_dir)

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
                    im_id2, fid = tracks[idx], tracks[idx + 1]
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
                cv2.circle(image, (x2 + image1.shape[1], y2), 20, (128, 128, 0), -1)
                cv2.line(image, (x1, y1), (x2 + image1.shape[1], y2), (0, 0, 0), 5)

            image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
            cv2.imwrite("debug/gt.jpg", image)
            cv2.imshow("", image)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def localize(self):
        return


if __name__ == '__main__':
    system = Localization()
    system.read_2d_2d_matches("query.jpg")