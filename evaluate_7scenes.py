import pathlib
import pickle
import sys
import gc
import cv2
import h5py
import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from tqdm import tqdm

import colmap_io
import retrieval_based_pycolmap
from math_utils import (geometric_verify_pydegensac, filter_pairs, quadrilateral_self_intersect_test)
from retrieval_utils import (CandidatePool, MatchCandidate,
                             log_matching, extract_global_descriptors_on_database_images, verify_matches_cross_compare)
from vis_utils import (visualize_cam_pose_with_point_cloud, visualize_matching_helper_with_pid2features,
                       concat_images_different_sizes)
from colmap_read import rotmat2qvec

sys.path.append("Hierarchical-Localization")
sys.path.append("cnnimageretrieval-pytorch")

from visloc_pseudo_gt_limitations import evaluation_util as vis_loc_pseudo_eval_utils
from torch.utils.model_zoo import load_url
from cirtorch.networks.imageretrievalnet import init_network

from pathlib import Path
from feature_matching import run_d2_detector_on_folder

from hloc import extractors
from hloc import extract_features, match_features_bare

from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names, get_matches_wo_loading
from hloc.reconstruction import get_image_ids
from evaluation_utils import (WS_FOLDER, IMAGES_ROOT_FOLDER, DB_DIR, prepare, read_logs)


class Localization:
    # @profile
    def __init__(self, db_names, query_names):
        if COMPARE_TO_GT:
            gt_file = "visloc_pseudo_gt_limitations/pgt/sfm/7scenes/redkitchen_test.txt"
            self.pgt_poses = vis_loc_pseudo_eval_utils.read_pose_data(gt_file)

        self.db_im_names = db_names
        self.query_im_names = query_names
        self.workspace_dir = WS_FOLDER
        self.workspace_images_dir = IMAGES_ROOT_FOLDER
        self.pose_result_file = f"{self.workspace_dir}/res_div.txt"

        self.workspace_database_dir = DB_DIR
        self.workspace_sfm_images_dir = f"{self.workspace_dir}/images.txt"
        self.workspace_sfm_point_cloud_dir = f"{self.workspace_dir}/points3D.txt"

        self.image_to_kp_tree = {}
        self.point3d_cloud = None
        self.image_name_to_image_id = {}
        self.image2pose = None
        self.image2pose_new = None
        self.pid2features = None  # maps pid to list of images that observes this point
        self.pid2features_rebuilt = False
        self.localization_results = []

        # matching database
        self.matches = None
        self.id2kp, self.id2desc, self.id2name, self.id2score = None, None, None, None
        self.h5_file_features = None
        self.query_desc_mat = None
        self.query_kp_mat = None
        self.d2_masks = None
        self.name2count = None
        self.most_common_images = []
        self.desc_tree = None
        self.pid2images = None
        self.point3did2xyzrgb = colmap_io.read_points3D_coordinates(self.workspace_sfm_point_cloud_dir)
        self.point_cloud = None
        self.default_metadata = {'f': 26.0, 'cx': 1134.0, 'cy': 2016.0}

        # image retrieval variables
        self.retrieval_images_dir = self.workspace_dir / 'images_retrieval'
        self.retrieval_loc_pairs_dir = self.workspace_dir / 'pairs.txt'  # top 20 retrieved by NetVLAD
        self.retrieval_conf = extract_features.confs['netvlad']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # cnn image retrieval
        training_weights = {
            '0': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
            '1': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth'
        }
        weights_folder = "cnnimageretrieval-pytorch/data/networks"
        self.state = load_url(training_weights['1'], model_dir=weights_folder)
        self.cnn_retrieval_net = init_network({'architecture': self.state['meta']['architecture'],
                                               'pooling': self.state['meta']['pooling'],
                                               'whitening': self.state['meta'].get('whitening')})
        self.cnn_retrieval_net.load_state_dict(self.state['state_dict'])
        self.cnn_retrieval_net.eval()
        self.cnn_retrieval_net.cuda()
        self.db_descriptors_dir = self.workspace_dir / "database_global_descriptors_0.pkl"
        if not self.db_descriptors_dir.exists():
            print("Global database descriptors not found, extracting ...")
            extract_global_descriptors_on_database_images(self.retrieval_images_dir / "db",
                                                          self.workspace_dir, multi_scale=False)

        # matching variables
        self.desc_type = None
        self.kp_type = None
        self.matching_conf = extract_features.confs['sift']
        model_class = dynamic_load(extractors, self.matching_conf['model']['name'])
        self.matching_model = model_class(self.matching_conf['model']).eval().to(self.device)
        self.matching_feature_path = Path(self.workspace_dir, self.matching_conf['output'] + '.h5')
        self.matching_feature_path.parent.mkdir(exist_ok=True, parents=True)
        self.matching_skip_names = set(list_h5_names(self.matching_feature_path)
                                       if self.matching_feature_path.exists() else ())
        self.name2ref = match_features_bare.return_name2ref(self.matching_feature_path)

        self.name2id = get_image_ids(self.workspace_database_dir)
        self.build_tree()
        self.reduce_memory(self.query_im_names)
        self.build_desc_tree()
        # self.reduce_memory(self.query_im_names)
        self.build_d2_masks()
        return

    def read_name2count(self):
        """
        maps im name to the number of times this im needs to be matched
        """
        sys.stdin = open(self.retrieval_loc_pairs_dir, "r")
        lines = sys.stdin.readlines()
        self.name2count = {}
        for line in lines:
            name = line[:-1].split(" ")[-1]
            if name not in self.name2count:
                self.name2count[name] = 1
            else:
                self.name2count[name] += 1
        self.most_common_images = sorted(list(self.name2count.keys()), key=lambda du: self.name2count[du],
                                         reverse=True)[:10]

    def terminate(self):
        if self.h5_file_features is not None:
            self.h5_file_features.close()
        self.h5_file_features = None
        self.query_desc_mat = None
        self.query_kp_mat = None

    def read_matches(self):
        with open(self.retrieval_loc_pairs_dir, 'r') as f:
            pairs = [p.split() for p in f.readlines()]
        image_ids = get_image_ids(self.workspace_database_dir)

        my_file = pathlib.Path(f"{WS_FOLDER}/matches_1.pkl")
        if my_file.is_file():
            with open(f"{WS_FOLDER}/matches_1.pkl", 'rb') as handle:
                self.matches = pickle.load(handle)
        else:
            self.matches = {}
            with h5py.File(str(self.workspace_dir / "matches.h5"), 'r') as hfile:
                for p1, p2 in tqdm(pairs, desc="Reading matches"):
                    m1, _ = get_matches_wo_loading(hfile, p1, p2)
                    key_ = image_ids[p1], image_ids[p2]
                    self.matches[key_] = m1
            with open(f"{WS_FOLDER}/matches_1.pkl", 'wb') as handle:
                pickle.dump(self.matches, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def read_features(self):
        """
        h5 file maps img name => ['descriptors', 'image_size', 'keypoints', 'scores']
        """
        self.h5_file_features = h5py.File(self.matching_feature_path, 'r')

    def reduce_memory(self, query_names):
        """
        only used in debug mode
        """
        sys.stdin = open(self.retrieval_loc_pairs_dir, "r")
        lines = sys.stdin.readlines()
        db_ids = []
        self.retrieval_pairs = {}
        for line in lines:
            line = line[:-1]
            q, db = line.split(" ")
            if self.name2id[q] not in self.retrieval_pairs:
                self.retrieval_pairs[self.name2id[q]] = [self.name2id[db]]
            else:
                self.retrieval_pairs[self.name2id[q]].append(self.name2id[db])
            if q in query_names and self.name2id[db] not in db_ids:
                db_ids.append(self.name2id[db])

        # delete
        all_ids = list(self.image2pose.keys())
        for im_id in all_ids:
            if im_id not in db_ids:
                if im_id in self.image2pose:
                    del self.image2pose[im_id]
                    del self.image_to_kp_tree[im_id]

        # if self.id2desc is not None:
        #     all_ids = list(self.id2desc.keys())
        #     for im_id in all_ids:
        #         if im_id not in db_ids:
        #             if im_id in self.id2desc:
        #                 del self.id2desc[im_id]
        gc.collect()

    # @profile
    def build_tree(self):
        self.read_name2count()

        my_file = pathlib.Path(f"{WS_FOLDER}/image2pose_by_id.pkl")
        if my_file.is_file():
            with open(f"{WS_FOLDER}/image2pose_by_id.pkl", 'rb') as handle:
                self.image2pose = pickle.load(handle)
        else:
            self.image2pose = colmap_io.read_images(self.workspace_sfm_images_dir)
            with open(f"{WS_FOLDER}/image2pose_by_id.pkl", 'wb') as handle:
                pickle.dump(self.image2pose, handle, protocol=pickle.HIGHEST_PROTOCOL)

        my_file = pathlib.Path(f"{WS_FOLDER}/pid2features.pkl")
        if my_file.is_file():
            with open(f"{WS_FOLDER}/pid2features.pkl", 'rb') as handle:
                self.pid2features = pickle.load(handle)
        else:
            self.pid2features = colmap_io.read_pid2images(self.image2pose)
            with open(f"{WS_FOLDER}/pid2features.pkl", 'wb') as handle:
                pickle.dump(self.pid2features, handle, protocol=pickle.HIGHEST_PROTOCOL)

        my_file = pathlib.Path(f"{WS_FOLDER}/image_to_kp_tree.pkl")
        if my_file.is_file():
            with open(f"{WS_FOLDER}/image_to_kp_tree.pkl", 'rb') as handle:
                self.image_to_kp_tree = pickle.load(handle)
        else:
            for img_id in self.image2pose:
                self.image_to_kp_tree[img_id] = []
            for img_id in self.image2pose:
                image_name, points2d_meaningful, cam_pose, cam_id = self.image2pose[img_id]
                # self.image_name_to_image_id[image_name] = img_id
                fid_list = []
                pid_list = []
                for fid, (fx, fy, pid) in enumerate(points2d_meaningful):
                    if pid >= 0:
                        fid_list.append(fid)
                        pid_list.append(pid)
                f_coord_list = np.zeros((len(fid_list), 2), dtype=np.float32)
                idx = 0
                for fx, fy, pid in points2d_meaningful:
                    if pid >= 0:
                        f_coord_list[idx, :] = [fx, fy]
                        idx += 1
                if image_name in self.most_common_images:
                    self.image_to_kp_tree[img_id] = (fid_list, pid_list, KDTree(f_coord_list), f_coord_list)
                else:
                    self.image_to_kp_tree[img_id] = (fid_list, pid_list, None, f_coord_list)
            with open(f"{WS_FOLDER}/image_to_kp_tree.pkl", 'wb') as handle:
                pickle.dump(self.image_to_kp_tree, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def build_d2_masks(self):
        """
        builds a mask telling which features are close to d2 features for all database images
        """
        self.read_features()
        self.d2_masks = {}
        self.id2kp = {}
        d2_file = run_d2_detector_on_folder(str(IMAGES_ROOT_FOLDER), str(self.workspace_dir),
                                            image_list=self.db_im_names)
        with open(d2_file, 'rb') as handle:
            name2kp = pickle.load(handle)
            for name in tqdm(self.name2id, desc="Building id2kp"):
                kp_mat = self.h5_file_features[name]["keypoints"].__array__()
                self.id2kp[name] = kp_mat

            for name in tqdm(self.db_im_names, desc="Building d2 masks"):
                img_id = self.name2id[name]
                kp_mat = self.h5_file_features[name]["keypoints"].__array__()
                kp_mat_d2 = name2kp[name]
                tree = KDTree(kp_mat_d2)
                dis_mat, idx_mat = tree.query(kp_mat, 1)
                mask = dis_mat < 5
                self.d2_masks[img_id] = (dis_mat, mask)
        self.terminate()

    def build_desc_tree(self):
        """
        builds a KD tree for all the descriptors of database images (for querying closest descriptors)
        """
        self.id2name = {}
        self.id2desc = {}
        with h5py.File(self.matching_feature_path, 'r') as hfile:
            self.desc_tree = {}
            for name in tqdm(self.name2id, desc="Building desc tree"):
                img_id = self.name2id[name]
                self.id2name[img_id] = name
                desc_mat = np.transpose(hfile[name]["descriptors"].__array__())
                self.id2desc[img_id] = desc_mat
                if name in self.most_common_images:
                    self.desc_tree[img_id] = KDTree(desc_mat)

    def get_feature_coord(self, img_id, fid):
        name = self.id2name[img_id]
        return self.id2kp[name][fid]

    def get_feature_desc(self, img_id, fid):
        return self.id2desc[img_id][fid, :]

    def get_feature_score(self, img_id, fid):
        name = self.id2name[img_id]
        return self.h5_file_features[name]["scores"][fid]

    def read_computed_poses(self):
        name2count = {}
        my_file = pathlib.Path(self.pose_result_file)
        if not my_file.is_file():
            return name2count
        sys.stdin = open(self.pose_result_file, "r")
        lines = sys.stdin.readlines()
        for line in lines:
            name = line[:-1].split(" ")[0]
            name2count[name] = line[:-1]
        return name2count

    def check_failed_results(self, names):
        computed_names = self.read_computed_poses()
        for name in names:
            if name not in computed_names:
                print(name)

    # @profile
    def main(self, metadata, name_list, re_write=False):
        error_changed = []
        self.read_matching_database()
        computed_names = self.read_computed_poses()
        if not DEBUG and not re_write and not COMPARE_TO_GT:
            name_list2 = [du for du in name_list if du not in computed_names]
        else:
            name_list2 = name_list
        for query_im_name in tqdm(name_list2, desc="Localizing"):
            tqdm.write(f"{query_im_name}")
            pairs, point3d_candidate_pool = self.read_2d_2d_matches(query_im_name, metadata)
            if len(pairs) <= 3:
                tqdm.write(f" localization failed (found {len(pairs)})")
                continue

            best_score = None
            best_pose = None
            best_diff = None
            for _ in range(10):
                r_mat, t_vec, score, mask, diff = retrieval_based_pycolmap.localize(metadata, pairs)
                if best_score is None or score > best_score:
                    best_score = score
                    best_pose = (r_mat, t_vec)
                    best_diff = diff
                    if best_score > 0.9:
                        break
            r_mat, t_vec = best_pose

            if DEBUG:
                with open(f"{self.workspace_dir}/logs.txt", "w") as a_file:
                    pgt_pose, rgb_focal_length = self.pgt_poses[query_im_name]
                    print(pgt_pose, file=a_file)
                    for xy, xyz in pairs:
                        print(f"{xy[0]} {xy[1]} {xyz[0]} {xyz[1]} {xyz[2]}", file=a_file)
                # candidates = point3d_candidate_pool.pool[:len(pairs)]
                # candidate_indices = sorted(list(range(len(pairs))), key=lambda du: best_diff[du])
                # for idx__, cand_idx in enumerate(candidate_indices):
                #     candidate = candidates[cand_idx]
                #     pid = candidate.pid
                #     xy = candidate.query_coord
                #     x2, y2 = map(int, xy)
                #     query_img = cv2.imread(f"{IMAGES_ROOT_FOLDER}/{query_im_name}")
                #     if query_img is None:
                #         print(f"{IMAGES_ROOT_FOLDER}/{query_im_name}")
                #         raise ValueError
                #     cv2.circle(query_img, (x2, y2), 10, (128, 128, 0), -1)
                #     vis_img = visualize_matching_helper_with_pid2features(query_img, self.pid2images[pid],
                #                                                           self.workspace_images_dir)
                #     cv2.imwrite(f"debug3/img-{idx__}-{round(best_diff[cand_idx], 5)}.jpg", vis_img)

            qw, qx, qy, qz = rotmat2qvec(r_mat)
            tx, ty, tz = t_vec[:, 0]
            result = f"{query_im_name} {qw} {qx} {qy} {qz} {tx} {ty} {tz}"

            if COMPARE_TO_GT:
                pgt_pose, rgb_focal_length = self.pgt_poses[query_im_name]
                est_poses = vis_loc_pseudo_eval_utils.convert_pose_data([result])
                est_pose, est_f = est_poses[query_im_name]
                error = vis_loc_pseudo_eval_utils.compute_error_max_rot_trans(pgt_pose, est_pose)
                if query_im_name in NAME2ERROR:
                    tqdm.write(f" error from {NAME2ERROR[query_im_name]} to {error} ")
                    error_changed.append(NAME2ERROR[query_im_name] - error)
                else:
                    tqdm.write(f" error from 0 to {error} ")

            if not DEBUG and not COMPARE_TO_GT and not re_write:
                with open(self.pose_result_file, "a") as a_file:
                    print(result, file=a_file)
            if re_write:
                self.name2count[query_im_name] = result
            self.localization_results.append(((r_mat, t_vec), (1, 0, 0)))
        if re_write:
            with open(f"{self.workspace_dir}/res_debug.txt", "w") as a_file:
                for query_im_name in computed_names:
                    result = computed_names[query_im_name]
                    print(result, file=a_file)
        self.terminate()
        if len(error_changed) > 0:
            print(error_changed, np.mean(error_changed))

    def visualize(self):
        self.prepare_visualization()
        visualize_cam_pose_with_point_cloud(self.point_cloud, self.localization_results)
        self.localization_results.clear()

    def read_matching_database(self):
        self.read_matches()
        self.read_features()

    def gather_matches(self, key_, arr, id1, query_im_id, database_im_id, filter_by_d2_detection=True):
        pairs = []
        points1 = []
        points2 = []
        all_pairs = []

        for u, v in arr:
            if id1 == database_im_id:
                database_fid = u
                query_fid = v
            else:
                database_fid = v
                query_fid = u

            database_fid_coord = self.get_feature_coord(database_im_id, database_fid)
            query_fid_coord = self.get_feature_coord(query_im_id, query_fid)  # hloc
            distance_to_d2_feature = None

            if DEBUG:
                all_pairs.append((query_fid_coord, database_fid_coord))

            if filter_by_d2_detection:
                # check if the matched feature of database image is close to one d2 feature
                dis_mat, mask = self.d2_masks[database_im_id]
                distance_to_d2_feature = dis_mat[database_fid]

            query_fid_desc = self.get_feature_desc(query_im_id, query_fid)  # hloc
            database_fid_desc = self.get_feature_desc(database_im_id, database_fid)  # hloc
            if database_im_id in self.desc_tree:
                dis, ind_list = self.desc_tree[database_im_id].query(query_fid_desc, 3)
            else:
                tree = KDTree(self.id2desc[database_im_id])
                dis, ind_list = tree.query(query_fid_desc, 3)

            ratio_test = dis[1] / dis[2]

            pair = (
                query_fid_coord, database_fid_coord, query_fid_desc, database_fid_desc, ratio_test,
                distance_to_d2_feature,
                self.id2name[database_im_id]
            )
            pairs.append(pair)
            points1.append(query_fid_coord)
            points2.append(database_fid_coord)
        if DEBUG:
            log_matching(all_pairs,
                         f"{IMAGES_ROOT_FOLDER}/{self.id2name[database_im_id]}",
                         f"{IMAGES_ROOT_FOLDER}/{self.id2name[query_im_id]}",
                         f"debug/{GLOBAL_COUNT}-1-all-pairs-{database_im_id}-{query_im_id}.jpg")
            log_matching(pairs,
                         f"{IMAGES_ROOT_FOLDER}/{self.id2name[database_im_id]}",
                         f"{IMAGES_ROOT_FOLDER}/{self.id2name[query_im_id]}",
                         f"debug/{GLOBAL_COUNT}-2-d2-pairs.jpg")
        return points1, points2, pairs

    # @profile
    def verify_matches_cross_compare(self, pairs, database_im_id_sfm, query_im_name):
        pairs2 = []
        debug_info = []
        for pair in pairs:
            query_fid_coord, database_fid_coord, query_fid_desc, database_fid_desc, ratio_test, distance_to_d2_feature, db_im_name = pair
            fid_list, pid_list, tree, f_coord_mat = self.image_to_kp_tree[database_im_id_sfm]  # sfm
            if tree is None:
                tree = KDTree(f_coord_mat)
            dis, ind = tree.query(database_fid_coord, 1)  # sfm
            pid = pid_list[ind]
            pairs2.append((np.array(query_fid_coord), pid, db_im_name))
            debug_info.append((f_coord_mat[ind], database_fid_coord, dis, db_im_name))
        query_kp_mat = self.h5_file_features[query_im_name]["keypoints"].__array__()
        query_im_id = self.name2id[query_im_name]
        mask2, scores2, totals2, logs_cc = verify_matches_cross_compare(self.matches, pairs2, self.pid2features,
                                                                        query_kp_mat,
                                                                        self.id2kp, query_im_id, self.name2id,
                                                                        debug_mode=DEBUG)
        if DEBUG:
            count = 0
            for query_fid_coord, fid2, database_fid_coord2, database_fid_coord3, query_id, name, pairs, idx_, key_, accept in logs_cc:
                database_fid_coord_pid, database_fid_coord, dis, db_im_name = debug_info[idx_]
                query_name = self.id2name[query_id]
                query_im = cv2.imread(f"{IMAGES_ROOT_FOLDER}/{query_name}")
                db_im = cv2.imread(f"{IMAGES_ROOT_FOLDER}/{name}")
                db_im0 = cv2.imread(f"{IMAGES_ROOT_FOLDER}/{db_im_name}")

                x, y = map(int, query_fid_coord)
                cv2.circle(query_im, (x, y), 10, (128, 0, 0), 2)
                x, y = map(int, database_fid_coord2)
                cv2.circle(db_im, (x, y), 10, (128, 0, 0), 2)
                x, y = map(int, database_fid_coord)
                cv2.circle(db_im0, (x, y), 10, (128, 0, 0), 2)
                x, y = map(int, database_fid_coord_pid)
                cv2.circle(db_im0, (x, y), 10, (128, 0, 0), -1)

                if accept == 0:
                    x, y = map(int, fid2)
                    cv2.circle(query_im, (x, y), 10, (0, 0, 255), 2)
                    x, y = map(int, database_fid_coord3)
                    cv2.circle(db_im, (x, y), 10, (0, 0, 255), 2)
                else:
                    x, y = map(int, fid2)
                    cv2.circle(query_im, (x, y), 10, (0, 255, 0), 2)
                    x, y = map(int, database_fid_coord3)
                    cv2.circle(db_im, (x, y), 10, (0, 255, 0), 2)
                image = concat_images_different_sizes([db_im0, query_im, db_im])
                count += 1
                cv2.imwrite(f"debug_cc/cc-{GLOBAL_COUNT}-{count}-{accept}-{dis}.jpg", image)

                log_matching(pairs,
                             f"{IMAGES_ROOT_FOLDER}/{name}",
                             f"{IMAGES_ROOT_FOLDER}/{query_name}",
                             f"debug_cc/cc-{GLOBAL_COUNT}-{count}-{query_im_id}-{self.name2id[name]}.jpg")
        return mask2, scores2, totals2

    def verify_matches(self, points1, points2, pairs, query_im_id, database_im_id, img_h, img_w, loose_control=True):
        if points1.shape[0] <= 3:  # too few matches
            if DEBUG:
                log_matching(pairs,
                             f"{IMAGES_ROOT_FOLDER}/{self.id2name[database_im_id]}",
                             f"{IMAGES_ROOT_FOLDER}/{self.id2name[query_im_id]}",
                             f"debug2/{GLOBAL_COUNT}-{points1.shape[0]}-few.jpg")
            return False, [False] * points1.shape[0]
        h_mat, mask, s1, s2 = geometric_verify_pydegensac(points1, points2)
        pairs = filter_pairs(pairs, mask)
        s2 = round(s2, 2)
        if np.sum(mask) == 0:  # homography is degenerate
            if DEBUG:
                log_matching(pairs,
                             f"{IMAGES_ROOT_FOLDER}/{self.id2name[database_im_id]}",
                             f"{IMAGES_ROOT_FOLDER}/{self.id2name[query_im_id]}",
                             f"debug2/{GLOBAL_COUNT}-{points1.shape[0]}-{s2}-mask0.jpg")
            return False, mask
        h, w = img_h, img_w
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, h_mat)
        dst = np.int64(dst)

        # normalize to center
        w3, h3 = np.min(dst[:, 0, 0]), np.min(dst[:, 0, 1])
        if w3 < 0:
            dst[:, 0, 0] -= w3
        if h3 < 0:
            dst[:, 0, 1] -= h3

        if DEBUG:
            plt.plot(
                [dst[0, 0, 0], dst[1, 0, 0]], [dst[0, 0, 1], dst[1, 0, 1]],
                [dst[1, 0, 0], dst[2, 0, 0]], [dst[1, 0, 1], dst[2, 0, 1]],
                [dst[2, 0, 0], dst[3, 0, 0]], [dst[2, 0, 1], dst[3, 0, 1]],
                [dst[3, 0, 0], dst[0, 0, 0]], [dst[3, 0, 1], dst[0, 0, 1]],
            )
            plt.savefig(f"debug2/{GLOBAL_COUNT}-shape.jpg")
            plt.close()
            img = cv2.imread(f"{IMAGES_ROOT_FOLDER}/{self.id2name[query_im_id]}")
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            img2_tr = cv2.polylines(img,
                                    [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imwrite(f"debug2/{GLOBAL_COUNT}-{points1.shape[0]}-{s2}-homo.jpg", img2_tr)

        if loose_control:
            return True, mask

        w2, h2 = np.max(dst[:, 0, 0]), np.max(dst[:, 0, 1])
        if max(w2, h2) > 10000:  # too large homography
            if DEBUG:
                log_matching(pairs,
                             f"{IMAGES_ROOT_FOLDER}/{self.id2name[database_im_id]}",
                             f"{IMAGES_ROOT_FOLDER}/{self.id2name[query_im_id]}",
                             f"debug2/{GLOBAL_COUNT}-{points1.shape[0]}-{s2}-large.jpg")
            return False, mask
        if w2 == 0 or h2 == 0:  # homography is degenerate
            if DEBUG:
                log_matching(pairs,
                             f"{IMAGES_ROOT_FOLDER}/{self.id2name[database_im_id]}",
                             f"{IMAGES_ROOT_FOLDER}/{self.id2name[query_im_id]}",
                             f"debug2/{GLOBAL_COUNT}-{points1.shape[0]}-{s2}-00.jpg")
            return False, mask

        # check if homography is degenerate
        self_intersect = quadrilateral_self_intersect_test(dst[0, 0, :], dst[1, 0, :],
                                                           dst[2, 0, :], dst[3, 0, :])
        if self_intersect:
            if DEBUG:
                log_matching(pairs,
                             f"{IMAGES_ROOT_FOLDER}/{self.id2name[database_im_id]}",
                             f"{IMAGES_ROOT_FOLDER}/{self.id2name[query_im_id]}",
                             f"debug2/{GLOBAL_COUNT}-{points1.shape[0]}-{s2}-intersect.jpg")
            return False, mask

        return True, mask

    def register_matches(self, pairs, database_im_id_sfm, point3d_candidate_pool, desc_heuristics):
        for pair in pairs:
            if len(pair) == 7:
                query_fid_coord, database_fid_coord, query_fid_desc, database_fid_desc, ratio_test, distance_to_d2_feature, _ = pair
                score = 1
            else:
                query_fid_coord, database_fid_coord, query_fid_desc, database_fid_desc, ratio_test, distance_to_d2_feature, _, score = pair
            if desc_heuristics:
                desc_diff = np.sqrt(np.sum(np.square(query_fid_desc - database_fid_desc))) / 128
            else:
                desc_diff = 1

            fid_list, pid_list, tree, f_coord_mat = self.image_to_kp_tree[database_im_id_sfm]  # sfm
            if tree is None:
                tree = KDTree(f_coord_mat)
            dis, ind = tree.query(database_fid_coord, 1)  # sfm

            pid = pid_list[ind]
            fid = fid_list[ind]
            candidate = MatchCandidate(query_fid_coord, fid, pid, dis, desc_diff, ratio_test, distance_to_d2_feature,
                                       score)
            point3d_candidate_pool.add(candidate)

    def read_2d_2d_matches(self, query_im_name, meta_data, max_nb_matches=500,
                           max_pool_size=500, cc_verification=False, sampled_by_collection=True):
        desc_heuristics = True
        query_im_id = self.name2id[query_im_name]
        all_matches = []
        all_matches_cc = []
        point3d_candidate_pool = CandidatePool()
        global GLOBAL_COUNT
        nb_skipped = 0
        total = 0
        sampled_matches = []
        if sampled_by_collection:
            collection2images = {}
            total_matches = 0
            for m in self.matches:
                if query_im_id in m:
                    arr = self.matches[m]
                    if arr is not None:
                        id1, id2 = m
                        if id1 != query_im_id:
                            database_im_id = id1
                        else:
                            database_im_id = id2
                        collection_name = self.id2name[database_im_id].split("/")[0]
                        total_matches += 1
                        if collection_name not in collection2images:
                            collection2images[collection_name] = [m]
                        else:
                            collection2images[collection_name].append(m)
            while len(sampled_matches) < total_matches:
                all_keys = list(collection2images.keys())
                collection_name = np.random.choice(all_keys)
                a_match = collection2images[collection_name].pop()
                sampled_matches.append(a_match)
                if len(collection2images[collection_name]) == 0:
                    del collection2images[collection_name]

        if sampled_matches:
            looping_array = sampled_matches
        else:
            looping_array = self.matches
        for m in looping_array:
            if len(point3d_candidate_pool) > max_nb_matches:
                break
            if query_im_id in m:
                arr = self.matches[m]
                if arr is not None:
                    GLOBAL_COUNT += 1
                    total += 1
                    id1, id2 = m
                    if id1 != query_im_id:
                        database_im_id = id1
                    else:
                        database_im_id = id2
                    database_im_id_sfm = database_im_id
                    if DEBUG:
                        tqdm.write(f" matching with {self.id2name[database_im_id_sfm]}")
                    original_points1, original_points2, original_pairs = self.gather_matches(m, arr, id1, query_im_id,
                                                                                             database_im_id,
                                                                                             filter_by_d2_detection=True)
                    if DEBUG:
                        tqdm.write(f" {GLOBAL_COUNT} gathering {len(original_pairs)} matches")
                    if len(original_pairs) == 0:
                        continue

                    if cc_verification:
                        mask0, scores, totals = self.verify_matches_cross_compare(original_pairs, database_im_id_sfm,
                                                                                  query_im_name)
                        points1, points2, pairs, scores = map(lambda du: filter_pairs(du, mask0),
                                                              [original_points1, original_points2,
                                                               original_pairs, scores])
                        pairs2 = []
                        for idx, pair in enumerate(pairs):
                            pair2 = list(pair)
                            pair2.append(scores[idx])
                            pairs2.append(pair2)
                        all_matches_cc.append((pairs2, database_im_id_sfm))
                    all_matches.append((original_pairs, database_im_id_sfm))

                    # homography ransac loop checking
                    points1_arr = np.vstack(original_points1)
                    points2_arr = np.vstack(original_points2)
                    verified, mask = self.verify_matches(points1_arr, points2_arr, original_pairs, query_im_id,
                                                         database_im_id,
                                                         meta_data["h"], meta_data["w"])
                    if not verified:
                        nb_skipped += 1
                        continue

                    points1, points2, pairs = map(lambda du: filter_pairs(du, mask),
                                                  [original_points1, original_points2, original_pairs])

                    if DEBUG:
                        tqdm.write(f" homography results in {len(pairs)} matches")
                        log_matching(pairs,
                                     f"{IMAGES_ROOT_FOLDER}/{self.id2name[database_im_id]}",
                                     f"{IMAGES_ROOT_FOLDER}/{self.id2name[query_im_id]}",
                                     f"debug/{GLOBAL_COUNT}-3-homo-pairs.jpg")

                    # cross compare
                    if cc_verification:
                        mask0, scores, totals = self.verify_matches_cross_compare(pairs, database_im_id_sfm,
                                                                                  query_im_name)
                        points1, points2, pairs, scores = map(lambda du: filter_pairs(du, mask0),
                                                              [points1, points2, pairs, scores])
                        if len(pairs) == 0:
                            continue
                        if DEBUG:
                            tqdm.write(f" cross comparing results in {len(pairs)} matches")
                            log_matching(pairs,
                                         f"{IMAGES_ROOT_FOLDER}/{self.id2name[database_im_id]}",
                                         f"{IMAGES_ROOT_FOLDER}/{self.id2name[query_im_id]}",
                                         f"debug/{GLOBAL_COUNT}-4-cc-pairs.jpg")

                        pairs2 = []
                        for idx, pair in enumerate(pairs):
                            pair2 = list(pair)
                            pair2.append(scores[idx])
                            pairs2.append(pair2)
                        self.register_matches(pairs2, database_im_id_sfm, point3d_candidate_pool, desc_heuristics)
                    else:
                        self.register_matches(pairs, database_im_id_sfm, point3d_candidate_pool, desc_heuristics)

        tqdm.write(f" have {len(point3d_candidate_pool)} matches")

        if len(point3d_candidate_pool) == 0:
            if len(all_matches_cc) > 5:
                tqdm.write(f" register cc matches")
                for p, db_id in all_matches_cc:
                    self.register_matches(p, db_id, point3d_candidate_pool, True)
            else:
                tqdm.write(f" register un filtered matches")
                for p, db_id in all_matches:
                    self.register_matches(p, db_id, point3d_candidate_pool, True)

        pairs = []
        self.pid2images = colmap_io.read_pid2images(self.image2pose)
        debug_info = [self.pid2images, IMAGES_ROOT_FOLDER, self.workspace_images_dir, query_im_name]
        if len(point3d_candidate_pool) > 0:
            point3d_candidate_pool.count_votes()
            point3d_candidate_pool.filter(debug_info, self.point3did2xyzrgb)
            point3d_candidate_pool.sort(by_votes=True)
            ratio = point3d_candidate_pool.divide()
            tqdm.write(f" ratio={ratio}")

            for cand_idx, candidate in enumerate(point3d_candidate_pool.pool[:max_pool_size]):
                pid = candidate.pid
                xy = candidate.query_coord
                xyz = self.point3did2xyzrgb[pid][:3]
                pairs.append((xy, xyz))

                if DEBUG:
                    # print(cand_idx, point3d_candidate_pool.pid2votes[candidate.pid],
                    #       candidate.desc_diff,
                    #       candidate.ratio_test,
                    #       candidate.d2_distance,
                    #       candidate.cc_score)
                    x2, y2 = map(int, xy)
                    query_img = cv2.imread(f"{IMAGES_ROOT_FOLDER}/{query_im_name}")
                    if query_img is None:
                        print(f"{IMAGES_ROOT_FOLDER}/{query_im_name}")
                        raise ValueError
                    cv2.circle(query_img, (x2, y2), 20, (128, 128, 0), 10)
                    vis_img = visualize_matching_helper_with_pid2features(query_img, self.pid2images[pid],
                                                                          self.workspace_images_dir)
                    cv2.imwrite(f"debug/img-{cand_idx}.jpg", vis_img)
        else:
            tqdm.write(" no matches found")

        return pairs, point3d_candidate_pool

    def prepare_visualization(self):
        points_3d_list = []

        for point3d_id in self.point3did2xyzrgb:
            x, y, z, r, g, b = self.point3did2xyzrgb[point3d_id]
            points_3d_list.append([x, y, z, r / 255, g / 255, b / 255])

        points_3d_list = np.vstack(points_3d_list)
        self.point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d_list[:, :3]))
        self.point_cloud.colors = o3d.utility.Vector3dVector(points_3d_list[:, 3:])
        self.point_cloud, _ = self.point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)


if __name__ == '__main__':
    DEBUG = True
    COMPARE_TO_GT = True
    GLOBAL_COUNT = 0
    NAME2ERROR = {}
    cam_mat = {'f': 525.505 / 100, 'cx': 320.0, 'cy': 240.0, "h": 640, "w": 480}
    query_image_names, database_image_names = prepare()

    logs = read_logs("visloc_pseudo_gt_limitations/ar_vloc_diversified_bad.txt")
    query_image_names = [du[0] for du in logs]
    NAME2ERROR = {du1: du2 for du1, du2 in logs}
    query_image_names = query_image_names[:1]
    localizer = Localization(database_image_names, query_image_names)
    localizer.main(cam_mat, query_image_names, re_write=False)
