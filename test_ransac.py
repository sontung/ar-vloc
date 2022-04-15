import cv2
import numpy as np

import colmap_db_read
import vis_utils
import matplotlib.pyplot as plt
import pydegensac
from time import time
from copy import deepcopy


def verify_pydegensac(src_pts, dst_pts, th = 4.0,  n_iter = 2000):
    H, mask = pydegensac.findHomography(src_pts, dst_pts, th, 0.99, n_iter)
    # print(f'pydegensac found {int(deepcopy(mask).astype(np.float32).sum())}/{src_pts.shape[0]} inliers')
    return H, mask, int(deepcopy(mask).astype(np.float32).sum()), deepcopy(mask).astype(np.float32).sum()/src_pts.shape[0]


def filter_pairs(ori_pairs, mask_):
    return [ori_pairs[idx] for idx in range(len(ori_pairs)) if mask_[idx]]


retrieval_images_dir = "/home/sontung/work/ar-vloc/vloc_workspace_retrieval/images_retrieval"
database_dir = "/home/sontung/work/ar-vloc/vloc_workspace_retrieval/database_hloc.db"
matches = colmap_db_read.extract_colmap_matches(database_dir)

id2kp, id2desc, id2name = colmap_db_read.extract_colmap_hloc(database_dir)

name2id = {name: ind for ind, name in id2name.items()}
desc_heuristics = False
query_im_name = "query/query.jpg"
query_im_id = name2id[query_im_name]
mean_distances = []
for m in matches:
    if query_im_id in m:
        arr = matches[m]
        if arr is not None:
            id1, id2 = m
            if id1 != query_im_id:
                database_im_id = id1
            else:
                database_im_id = id2
            kp_dict = {id1: [], id2: []}

            pairs = []
            points1 = []
            points2 = []
            for u, v in arr:
                kp_dict[id1].append(u)
                kp_dict[id2].append(v)
                database_fid = kp_dict[database_im_id][-1]
                query_fid = kp_dict[query_im_id][-1]

                database_fid_coord = id2kp[database_im_id][database_fid]  # hloc
                query_fid_coord = id2kp[query_im_id][query_fid]  # hloc
                query_fid_desc = None

                x1, y1 = map(int, query_fid_coord)
                x2, y2 = map(int, database_fid_coord)

                pair = ((x1, y1), (x2, y2))
                pairs.append(pair)
                points1.append(query_fid_coord)
                points2.append(database_fid_coord)
            points1 = np.vstack(points1)
            points2 = np.vstack(points2)
            h_mat, mask, s1, s2 = verify_pydegensac(points1, points2)

            db_img = cv2.imread(f"{retrieval_images_dir}/{id2name[database_im_id]}")
            query_img = cv2.imread(f"{retrieval_images_dir}/{id2name[query_im_id]}")

            h, w, ch = query_img.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, h_mat)
            db_img = cv2.polylines(db_img, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
            img = vis_utils.visualize_matching_pairs(query_img, db_img, pairs)
            # img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
            # cv2.imshow("", img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            name = id2name[database_im_id].split("/")[-1]
            cv2.imwrite(f"debug2/{name}-0", img)

            # pairs ransac
            img = vis_utils.visualize_matching_pairs(query_img, db_img, filter_pairs(pairs, mask))
            name = id2name[database_im_id].split("/")[-1]
            cv2.imwrite(f"debug2/{name}-2-{s1}-{s2}.jpg", img)

matches, _ = colmap_db_read.extract_colmap_two_view_geometries(database_dir)
for m in matches:
    if query_im_id in m:
        arr = matches[m]
        if arr is not None:
            id1, id2 = m
            if id1 != query_im_id:
                database_im_id = id1
            else:
                database_im_id = id2
            kp_dict = {id1: [], id2: []}

            pairs = []
            points1 = []
            points2 = []
            for u, v in arr:
                kp_dict[id1].append(u)
                kp_dict[id2].append(v)
                database_fid = kp_dict[database_im_id][-1]
                query_fid = kp_dict[query_im_id][-1]

                database_fid_coord = id2kp[database_im_id][database_fid]  # hloc
                query_fid_coord = id2kp[query_im_id][query_fid]  # hloc
                query_fid_desc = None

                x1, y1 = map(int, query_fid_coord)
                x2, y2 = map(int, database_fid_coord)

                pair = ((x1, y1), (x2, y2))
                pairs.append(pair)
                points1.append(query_fid_coord)
                points2.append(database_fid_coord)
            points1 = np.vstack(points1)
            points2 = np.vstack(points2)
            h_mat, mask, s1, s2 = verify_pydegensac(points1, points2)

            db_img = cv2.imread(f"{retrieval_images_dir}/{id2name[database_im_id]}")
            query_img = cv2.imread(f"{retrieval_images_dir}/{id2name[query_im_id]}")

            h, w, ch = query_img.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, h_mat)
            db_img = cv2.polylines(db_img, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
            img = vis_utils.visualize_matching_pairs(query_img, db_img, pairs)
            name = id2name[database_im_id].split("/")[-1]
            cv2.imwrite(f"debug2/{name}-1-{s1}-{s2}.jpg", img)

        else:
            id1, id2 = m
            print(id2name[id1], id2name[id2])