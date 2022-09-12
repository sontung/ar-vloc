import pickle
import subprocess
import sys
import time
from os import listdir
from os.path import isfile, join

import cv2
import h5py
import numpy as np
import open3d as o3d
import torch

from scipy.spatial import KDTree
from tqdm import tqdm

import colmap_db_read
import colmap_io
import localization
from math_utils import (geometric_verify_pydegensac, filter_pairs, quadrilateral_self_intersect_test)
from retrieval_utils import (CandidatePool, MatchCandidate, extract_retrieval_pairs,
                             log_matching, verify_matches_cross_compare, verify_matches_cross_compare_fast,
                             extract_global_descriptors_on_database_images)
from vis_utils import (visualize_cam_pose_with_point_cloud, visualize_matching_helper_with_pid2features)

sys.path.append("Hierarchical-Localization")
sys.path.append("cnnimageretrieval-pytorch")

from pathlib import Path
from feature_matching import run_d2_detector_on_folder

from hloc import extractors
from hloc import extract_features, pairs_from_retrieval, match_features_bare

from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names, get_matches_wo_loading, get_matches_wo_loading_no_scores
from hloc.triangulation import (import_features)
from hloc.reconstruction import create_empty_db, import_images, get_image_ids

from torch.utils.model_zoo import load_url
from cirtorch.networks.imageretrievalnet import init_network
from multiprocessing import Process

DEBUG = False
GLOBAL_COUNT = 1


def localize(metadata, pairs):
    f = metadata["f"]
    cx = metadata["cx"]
    cy = metadata["cy"]

    r_mat, t_vec, score, mask = localization.localize_pose_lib(pairs, f, cx, cy)
    return r_mat, t_vec, score, mask, -1


def api_test():
    system = Localization(skip_geometric_verification=True)
    query_folder = "test_line_jpg"
    name_list = [f for f in listdir(query_folder) if isfile(join(query_folder, f))]
    query_folder_workspace1 = "vloc_workspace_retrieval/images"
    query_folder_workspace2 = "vloc_workspace_retrieval/images_retrieval/query"
    default_metadata = {'f': 26.0, 'cx': 1134.0, 'cy': 2016.0}

    start = time.time()
    # name_list = ["line-14.jpg"]
    # name_list = ["line-12.jpg"]
    # name_list = ["line-9.jpg"]
    # name_list = name_list[:2]

    scores = []
    images_with_scores = []
    for name in name_list:
        print(f"Processing {name}")
        query_img = cv2.imread(f"{query_folder}/{name}")
        cv2.imwrite(f"{query_folder_workspace1}/query.jpg", query_img)
        cv2.imwrite(f"{query_folder_workspace2}/query.jpg", query_img)
        score = system.main_for_api(default_metadata)
        scores.append(score)
        images_with_scores.append((name, score))
        # system.visualize()
        # if len(system.localization_results) > 0:
        #     system.localization_results.pop()
        # break

    end = time.time()
    print(f"Done in {end - start} seconds, avg. {(end - start) / len(name_list)} s/image")
    print(f"Avg. inlier ratio = {np.mean(scores)}")
    system.visualize()
    images_with_scores = sorted(images_with_scores, key=lambda du: du[-1])
    for du in images_with_scores:
        print(du)


class Localization:
    def __init__(self, skip_geometric_verification=False):
        self.all_queries_folder = "Test line"
        self.workspace_dir = Path('vloc_workspace_retrieval')
        self.workspace_images_dir = f"{self.workspace_dir}/images"
        self.workspace_loc_dir = f"{self.workspace_dir}/new"

        self.workspace_database_dir = self.workspace_dir / "database_hloc.db"
        self.workspace_queries_dir = f"{self.workspace_dir}/query"
        self.workspace_existing_model = f"{self.workspace_dir}/sparse"
        self.workspace_sfm_images_dir = f"{self.workspace_existing_model}/images.txt"
        self.workspace_sfm_point_cloud_dir = f"{self.workspace_existing_model}/points3D.txt"

        self.image_to_kp_tree = {}
        self.point3d_cloud = None
        self.image_name_to_image_id = {}
        self.image2pose = None
        self.image2pose_new = None
        self.pid2features = None  # maps pid to list of images that observes this point
        self.pid2features_rebuilt = False
        self.pid2names = None
        self.localization_results = []
        self.build_tree()

        # matching database
        self.matches = None
        self.id2kp, self.id2desc, self.id2name, self.id2score = None, None, None, None
        self.h5_file_features = None
        self.query_desc_mat = None
        self.query_kp_mat = None
        self.d2_masks = None

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
        model_class = dynamic_load(extractors, self.retrieval_conf['model']['name'])
        self.retrieval_model = model_class(self.retrieval_conf['model']).eval().to(self.device)
        self.feature_path = Path(self.workspace_dir, self.retrieval_conf['output'] + '.h5')
        self.feature_path.parent.mkdir(exist_ok=True, parents=True)
        self.skip_names = set(list_h5_names(self.feature_path) if self.feature_path.exists() else ())

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
        self.skip_geometric_verification = skip_geometric_verification
        self.matching_conf = extract_features.confs['sift']
        model_class = dynamic_load(extractors, self.matching_conf['model']['name'])
        self.matching_model = model_class(self.matching_conf['model']).eval().to(self.device)
        self.matching_feature_path = Path(self.workspace_dir, self.matching_conf['output'] + '.h5')
        self.matching_feature_path.parent.mkdir(exist_ok=True, parents=True)
        self.matching_skip_names = set(list_h5_names(self.matching_feature_path)
                                       if self.matching_feature_path.exists() else ())
        self.name2ref = match_features_bare.return_name2ref(self.matching_feature_path)
        self.matching_feature_file = None

        # pairs from retrieval variables
        db_descriptors = self.feature_path
        if isinstance(db_descriptors, (Path, str)):
            db_descriptors = [db_descriptors]
        name2db = {n: i for i, p in enumerate(db_descriptors)
                   for n in list_h5_names(p)}
        db_names_h5 = list(name2db.keys())
        query_names_h5 = list_h5_names(self.feature_path)
        self.db_names = pairs_from_retrieval.parse_names("db", None, db_names_h5)
        if len(self.db_names) == 0:
            raise ValueError('Could not find any database image.')

        if self.workspace_database_dir.exists():
            print("Precomputed database found")
        else:
            print("Precomputing database")
            self.create_hloc_database()

        self.name2id = get_image_ids(self.workspace_database_dir)
        self.query_names = pairs_from_retrieval.parse_names("query", None, query_names_h5)
        self.db_desc = pairs_from_retrieval.get_descriptors(self.db_names, db_descriptors, name2db)
        self.db_desc = self.db_desc.to(self.device)
        self.build_desc_tree()
        self.build_d2_masks()
        self.build_matching_feature_file()
        return

    def build_matching_feature_file(self):
        if self.matching_feature_file is None:
            self.matching_feature_file = {}
            fd_file = h5py.File(str(self.matching_feature_path), 'a')
            for name0 in self.name2id:
                grp = fd_file[name0]
                self.matching_feature_file[name0] = {}
                self.matching_feature_file[name0]["image_size"] = grp["image_size"]
                for k, v in grp.items():
                    self.matching_feature_file[name0][k] = torch.from_numpy(v.__array__()).float().to(self.device)
            fd_file.close()

    def create_hloc_database(self):
        """
        Run this when hloc database needs to be reset
        """
        create_empty_db(self.workspace_database_dir)
        import_images(self.retrieval_images_dir, self.workspace_database_dir, "PER_FOLDER", None)
        extract_features.main_wo_model_loading(self.matching_model, self.device, [],
                                               self.matching_conf, self.retrieval_images_dir,
                                               feature_path=self.matching_feature_path)
        image_ids = get_image_ids(self.workspace_database_dir)
        database_image_ids = {u: v for u, v in image_ids.items() if "db" in u}
        import_features(database_image_ids, self.workspace_database_dir, self.matching_feature_path)

    def run_feature_extraction(self):
        extract_features.main_wo_model_loading(self.matching_model, "cpu", self.matching_skip_names,
                                               self.matching_conf, self.retrieval_images_dir,
                                               feature_path=self.matching_feature_path)

    # @profile
    def run_image_retrieval_and_matching(self):
        extract_feature_process = Process(target=self.run_feature_extraction)
        extract_feature_process.start()

        # retrieve
        extract_retrieval_pairs(self.cnn_retrieval_net, self.state,
                                self.db_descriptors_dir,
                                self.retrieval_images_dir / "query/query.jpg",
                                self.workspace_dir / "retrieval_pairs.txt",
                                multi_scale=False,
                                nb_neighbors=40)

        # match
        extract_feature_process.join()
        extract_feature_process.kill()

        matching_conf = match_features_bare.confs["NN-ratio"]
        match_features_bare.main(self.name2ref, matching_conf, self.retrieval_loc_pairs_dir, self.matching_feature_path,
                                 matches=self.workspace_dir / "matches.h5",
                                 matching_feature_file=self.matching_feature_file, overwrite=True)

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
        self.matches = {}
        with h5py.File(str(self.workspace_dir / "matches.h5"), 'r') as hfile:
            for p1, p2 in pairs:
                m1 = get_matches_wo_loading_no_scores(hfile, p1, p2)
                key_ = image_ids[p1], image_ids[p2]
                self.matches[key_] = m1

    def read_features(self):
        """
        h5 file maps img name => ['descriptors', 'image_size', 'keypoints', 'scores']
        """
        self.h5_file_features = h5py.File(self.matching_feature_path, 'r')

    def build_tree(self):
        self.image2pose = colmap_io.read_images(self.workspace_sfm_images_dir)
        self.pid2features = colmap_io.read_pid2images(self.image2pose)
        self.pid2names = {}
        for pid in self.pid2features:
            self.pid2names[pid] = [du[1] for du in self.pid2features[pid]]

        for img_id in self.image2pose:
            self.image_to_kp_tree[img_id] = []
        for img_id in self.image2pose:
            image_name, points2d_meaningful, cam_pose, cam_id = self.image2pose[img_id]
            self.image_name_to_image_id[image_name] = img_id
            fid_list = []
            pid_list = []
            f_coord_list = []
            for fid, (fx, fy, pid) in enumerate(points2d_meaningful):
                if pid >= 0:
                    fid_list.append(fid)
                    pid_list.append(pid)
                    f_coord_list.append([fx, fy])
            self.image_to_kp_tree[img_id] = (fid_list, pid_list, KDTree(np.array(f_coord_list)), f_coord_list)

    def rebuild_pid2features(self):
        """
        re-build with new format for faster performance
        """
        assert len(self.name2id) > 0
        query_img_id = self.name2id["query/query.jpg"]
        if not self.pid2features_rebuilt:
            for pid in self.pid2features:
                new_data = []
                for img_id2, name, cx, cy in self.pid2features[pid]:
                    name = f"db/{name}"
                    if name in self.name2id:
                        img_id = self.name2id[name]
                        key1 = (query_img_id, img_id)
                        key2 = (img_id, query_img_id)
                        database_fid_coord = np.array([cx, cy], dtype=np.float16)
                        new_data.append([img_id, name, database_fid_coord, key1, key2])
                self.pid2features[pid] = new_data
            self.pid2features_rebuilt = True

    def build_d2_masks(self):
        """
        builds a mask telling which features are close to d2 features for all database images
        """
        self.read_features()
        self.d2_masks = {}
        self.id2kp = {}
        d2_file = run_d2_detector_on_folder(str(self.workspace_dir / "images_retrieval/db"), str(self.workspace_dir))
        with open(d2_file, 'rb') as handle:
            name2kp = pickle.load(handle)
            for name in self.name2id:
                if "query" not in name:
                    img_id = self.name2id[name]
                    kp_mat = self.h5_file_features[name]["keypoints"].__array__()
                    self.id2kp[img_id] = kp_mat
                    kp_mat_d2 = name2kp[name.split("/")[-1]]
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
            if self.desc_tree is None:
                self.desc_tree = {}
                for name in self.name2id:
                    img_id = self.name2id[name]
                    self.id2name[img_id] = name
                    if "query" not in name:
                        desc_mat = np.transpose(hfile[name]["descriptors"].__array__())
                        self.id2desc[img_id] = desc_mat
                        self.desc_tree[img_id] = KDTree(desc_mat)
        self.rebuild_pid2features()

    def get_feature_coord(self, img_id, fid, db=False):
        if db:
            return self.id2kp[img_id][fid]
        else:
            name = self.id2name[img_id]
            assert name == "query/query.jpg"
            if self.query_kp_mat is None:
                self.query_kp_mat = self.h5_file_features[name]["keypoints"].__array__()
        return self.query_kp_mat[fid]

    def get_feature_desc(self, img_id, fid, db=False):
        if db:
            return self.id2desc[img_id][fid, :]
        else:
            name = self.id2name[img_id]
            assert name == "query/query.jpg"
            if self.query_desc_mat is None:
                self.query_desc_mat = self.h5_file_features[name]["descriptors"].__array__()
        return self.query_desc_mat[:, fid]

    def get_feature_score(self, img_id, fid):
        name = self.id2name[img_id]
        return self.h5_file_features[name]["scores"][fid]

    # @profile
    def main_for_api(self, metadata):
        name_list = [f for f in listdir(self.workspace_queries_dir) if isfile(join(self.workspace_queries_dir, f))]
        self.run_image_retrieval_and_matching()
        self.read_matching_database()
        query_im_name = f"query/{name_list[0]}"
        pairs, point3d_candidate_pool = self.read_2d_2d_matches(query_im_name, debug=DEBUG)
        if len(pairs) == 0:
            return 0, None

        best_score = None
        best_pose = None
        best_mask = None
        for _ in range(10):
            r_mat, t_vec, score, mask = localize(metadata, pairs)
            if best_score is None or score > best_score:
                best_score = score
                best_pose = (r_mat, t_vec)
                best_mask = mask
                if best_score > 0.9:
                    break
        r_mat, t_vec = best_pose

        for cand_idx, candidate in enumerate(point3d_candidate_pool.pool[:len(pairs)]):
            pid = candidate.pid
            xy = candidate.query_coord
            xyz = self.point3did2xyzrgb[pid][:3]
            pairs.append((xy, xyz))

            if DEBUG and best_mask[cand_idx]:
                self.pid2images = colmap_io.read_pid2images(self.image2pose)
                print(cand_idx, point3d_candidate_pool.pid2votes[candidate.pid],
                      candidate.desc_diff,
                      candidate.ratio_test,
                      candidate.d2_distance,
                      candidate.cc_score)
                x2, y2 = map(int, xy)
                query_img = cv2.imread(f"{self.retrieval_images_dir}/{query_im_name}")
                if query_img is None:
                    print(f"{self.retrieval_images_dir}/{query_im_name}")
                    raise ValueError
                cv2.circle(query_img, (x2, y2), 50, (128, 128, 0), 10)
                vis_img = visualize_matching_helper_with_pid2features(query_img, self.pid2images[pid],
                                                                      self.workspace_images_dir)
                cv2.imwrite(f"debug3/img-{cand_idx}.jpg", vis_img)

        self.localization_results.append(((r_mat, t_vec), (1, 0, 0)))
        self.terminate()
        return best_score

    def visualize(self):
        self.prepare_visualization()
        visualize_cam_pose_with_point_cloud(self.point_cloud, self.localization_results)
        self.localization_results.clear()

    def read_matching_database(self):
        self.read_matches()
        self.read_features()

    def gather_matches(self, arr, id1, query_im_id, database_im_id, filter_by_d2_detection=True):
        pairs = []
        points1 = []
        points2 = []
        for u, v in arr:
            if id1 == database_im_id:
                database_fid = u
                query_fid = v
            else:
                database_fid = v
                query_fid = u

            database_fid_coord = self.get_feature_coord(database_im_id, database_fid, db=True)
            query_fid_coord = self.get_feature_coord(query_im_id, query_fid)  # hloc
            distance_to_d2_feature = None

            if filter_by_d2_detection:
                # check if the matched feature of database image is close to one d2 feature
                dis_mat, mask = self.d2_masks[database_im_id]
                close_to_a_d2_feature = mask[database_fid]
                if not close_to_a_d2_feature:
                    continue
                distance_to_d2_feature = dis_mat[database_fid]

            query_fid_desc = self.get_feature_desc(query_im_id, query_fid)  # hloc
            database_fid_desc = self.get_feature_desc(database_im_id, database_fid, db=True)  # hloc
            dis, ind_list = self.desc_tree[database_im_id].query(query_fid_desc, 3)
            ratio_test = dis[1] / dis[2]

            pair = (
                query_fid_coord, database_fid_coord, query_fid_desc, database_fid_desc, ratio_test,
                distance_to_d2_feature,
                self.id2name[database_im_id]
            )
            pairs.append(pair)
            points1.append(query_fid_coord)
            points2.append(database_fid_coord)
        return points1, points2, pairs

    # @profile
    def verify_matches_cross_compare(self, pairs, database_im_id_sfm):
        pairs2 = []
        for pair in pairs:
            query_fid_coord, database_fid_coord, query_fid_desc, database_fid_desc, ratio_test, distance_to_d2_feature, db_im_name = pair
            fid_list, pid_list, tree, _ = self.image_to_kp_tree[database_im_id_sfm]  # sfm
            dis, ind = tree.query(database_fid_coord, 1)  # sfm
            pid = pid_list[ind]
            pairs2.append((np.array(query_fid_coord), pid, db_im_name))
        if self.query_kp_mat is None:
            self.query_kp_mat = self.h5_file_features["query/query.jpg"]["keypoints"].__array__()
        mask2, scores2 = verify_matches_cross_compare_fast(self.matches, pairs2, self.pid2features, self.query_kp_mat,
                                                           self.id2kp, self.name2id)
        return mask2, scores2

    def verify_matches(self, points1, points2, pairs, query_im_id, database_im_id):
        if points1.shape[0] < 10:  # too few matches
            if DEBUG:
                log_matching(pairs,
                             f"{self.retrieval_images_dir}/{self.id2name[database_im_id]}",
                             f"{self.retrieval_images_dir}/{self.id2name[query_im_id]}",
                             f"debug2/{GLOBAL_COUNT}-{points1.shape[0]}-few.jpg")
            return True, [True] * points1.shape[0]
        h_mat, mask, s1, s2 = geometric_verify_pydegensac(points1, points2)
        pairs = filter_pairs(pairs, mask)
        s2 = round(s2, 2)
        if np.sum(mask) == 0:  # homography is degenerate
            if DEBUG:
                log_matching(pairs,
                             f"{self.retrieval_images_dir}/{self.id2name[database_im_id]}",
                             f"{self.retrieval_images_dir}/{self.id2name[query_im_id]}",
                             f"debug2/{GLOBAL_COUNT}-{points1.shape[0]}-{s2}-mask0.jpg")
            return False, mask
        h, w = 4032, 2268
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, h_mat)
        dst = np.int64(dst)

        # normalize to center
        w3, h3 = np.min(dst[:, 0, 0]), np.min(dst[:, 0, 1])
        if w3 < 0:
            dst[:, 0, 0] -= w3
        if h3 < 0:
            dst[:, 0, 1] -= h3

        w2, h2 = np.max(dst[:, 0, 0]), np.max(dst[:, 0, 1])
        if max(w2, h2) > 10000:  # too large homography
            if DEBUG:
                log_matching(pairs,
                             f"{self.retrieval_images_dir}/{self.id2name[database_im_id]}",
                             f"{self.retrieval_images_dir}/{self.id2name[query_im_id]}",
                             f"debug2/{GLOBAL_COUNT}-{points1.shape[0]}-{s2}-large.jpg")
            return False, mask
        if w2 == 0 or h2 == 0:  # homography is degenerate
            if DEBUG:
                log_matching(pairs,
                             f"{self.retrieval_images_dir}/{self.id2name[database_im_id]}",
                             f"{self.retrieval_images_dir}/{self.id2name[query_im_id]}",
                             f"debug2/{GLOBAL_COUNT}-{points1.shape[0]}-{s2}-00.jpg")
            return False, mask

        # check if homography is degenerate
        self_intersect = quadrilateral_self_intersect_test(dst[0, 0, :], dst[1, 0, :],
                                                           dst[2, 0, :], dst[3, 0, :])
        if self_intersect:
            if DEBUG:
                log_matching(pairs,
                             f"{self.retrieval_images_dir}/{self.id2name[database_im_id]}",
                             f"{self.retrieval_images_dir}/{self.id2name[query_im_id]}",
                             f"debug2/{GLOBAL_COUNT}-{points1.shape[0]}-{s2}-intersect.jpg")
            return False, mask

        return True, mask

    def register_matches(self, pairs, database_im_id_sfm, point3d_candidate_pool, desc_heuristics):
        for pair in pairs:
            query_fid_coord, database_fid_coord, query_fid_desc, database_fid_desc, ratio_test, distance_to_d2_feature, _, score = pair
            if desc_heuristics:
                desc_diff = np.sqrt(np.sum(np.square(query_fid_desc - database_fid_desc))) / 128
            else:
                desc_diff = 1
            fid_list, pid_list, tree, _ = self.image_to_kp_tree[database_im_id_sfm]  # sfm
            dis, ind = tree.query(database_fid_coord, 1)  # sfm
            pid = pid_list[ind]
            fid = fid_list[ind]
            candidate = MatchCandidate(query_fid_coord, fid, pid, dis, desc_diff, ratio_test, distance_to_d2_feature,
                                       score)
            point3d_candidate_pool.add(candidate)

    def read_2d_2d_matches(self, query_im_name, debug=False, max_pool_size=100):
        name2id = {name: ind for ind, name in self.id2name.items()}
        desc_heuristics = True
        query_im_id = name2id[query_im_name]
        point3d_candidate_pool = CandidatePool()
        global GLOBAL_COUNT
        nb_skipped = 0
        using = 0
        total = 0
        for m in self.matches:
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
                    database_im_id_sfm = self.image_name_to_image_id[self.id2name[database_im_id].split("/")[-1]]

                    points1, points2, pairs = self.gather_matches(arr, id1, query_im_id,
                                                                  database_im_id, filter_by_d2_detection=True)
                    mask0, scores = self.verify_matches_cross_compare(pairs, database_im_id_sfm)
                    points1, points2, pairs, scores = map(lambda du: filter_pairs(du, mask0),
                                                          [points1, points2, pairs, scores])
                    print(f" cross comparing reduces {len(mask0)} matches to {len(pairs)} matches")
                    if len(pairs) == 0:
                        continue
                    points1 = np.vstack(points1)
                    points2 = np.vstack(points2)

                    # homography ransac loop checking
                    verified, mask = self.verify_matches(points1, points2, pairs, query_im_id, database_im_id)
                    if not verified:
                        nb_skipped += 1
                        continue

                    scores = filter_pairs(scores, mask)
                    pairs = filter_pairs(pairs, mask)
                    pairs2 = []
                    for idx, pair in enumerate(pairs):
                        pair2 = list(pair)
                        pair2.append(scores[idx])
                        pairs2.append(pair2)

                    using += 1
                    if DEBUG:
                        log_matching(pairs,
                                     f"{self.retrieval_images_dir}/{self.id2name[database_im_id]}",
                                     f"{self.retrieval_images_dir}/{self.id2name[query_im_id]}",
                                     f"debug2/{GLOBAL_COUNT}-{points1.shape[0]}-useful.jpg")
                    self.register_matches(pairs2, database_im_id_sfm, point3d_candidate_pool, desc_heuristics)

        print(f" found {total} pairs, skipped {nb_skipped} pairs, used {using} pairs")
        if len(point3d_candidate_pool) > 0:
            point3d_candidate_pool.count_votes()
            point3d_candidate_pool.filter()
            point3d_candidate_pool.sort(by_votes=True)
        pairs = []
        for cand_idx, candidate in enumerate(point3d_candidate_pool.pool[:max_pool_size]):
            pid = candidate.pid
            xy = candidate.query_coord
            xyz = self.point3did2xyzrgb[pid][:3]
            pairs.append((xy, xyz))

            if DEBUG:
                self.pid2images = colmap_io.read_pid2images(self.image2pose)
                print(cand_idx, point3d_candidate_pool.pid2votes[candidate.pid],
                      candidate.desc_diff,
                      candidate.ratio_test,
                      candidate.d2_distance,
                      candidate.cc_score)
                x2, y2 = map(int, xy)
                query_img = cv2.imread(f"{self.retrieval_images_dir}/{query_im_name}")
                if query_img is None:
                    print(f"{self.retrieval_images_dir}/{query_im_name}")
                    raise ValueError
                cv2.circle(query_img, (x2, y2), 50, (128, 128, 0), 10)
                vis_img = visualize_matching_helper_with_pid2features(query_img, self.pid2images[pid],
                                                                      self.workspace_images_dir)
                cv2.imwrite(f"debug/img-{cand_idx}.jpg", vis_img)

        tqdm.write(f" voting-based filters {len(pairs)} pairs from {len(point3d_candidate_pool.pool)}")
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
    api_test()

    # system = Localization()
    # # system.create_hloc_database()
    # copy_process = system.create_folders()
    # copy_process.wait()
    # copy_process.kill()
    # system.run_image_retrieval_and_matching()
    # system.read_matching_database(system.workspace_database_dir)
    # for img_id in system.id2kp:
    #     print(img_id)
