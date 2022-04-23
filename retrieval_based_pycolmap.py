import subprocess
import sys
import time
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import open3d as o3d
import torch
import h5py
from scipy.spatial import KDTree
from tqdm import tqdm

import colmap_db_read
import colmap_io
import localization
from math_utils import (geometric_verify_pydegensac, filter_pairs, quadrilateral_self_intersect_test)
from retrieval_utils import (CandidatePool, MatchCandidate, extract_retrieval_pairs, log_matching)
from vis_utils import (visualize_cam_pose_with_point_cloud, visualize_matching_helper_with_pid2features)

sys.path.append("Hierarchical-Localization")
sys.path.append("cnnimageretrieval-pytorch")


from pathlib import Path
from hloc import extractors
from hloc import extract_features, pairs_from_retrieval, match_features_bare

from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names, get_matches, get_matches_wo_loading
from hloc.triangulation import (import_features, import_matches, geometric_verification)
from hloc.reconstruction import create_empty_db, import_images, get_image_ids

from torch.utils.model_zoo import load_url
from cirtorch.networks.imageretrievalnet import init_network, extract_ms, extract_ss

DEBUG = False
GLOBAL_COUNT = 1


def localize(metadata, pairs):
    f = metadata["f"] * 100
    cx = metadata["cx"]
    cy = metadata["cy"]

    r_mat, t_vec, score = localization.localize_single_image_lt_pnp(pairs, f, cx, cy, with_inliers_percent=True)
    return r_mat, t_vec, score


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
    # name_list = name_list[:5]

    scores = []
    images_with_scores = []
    for name in name_list:
        print(name)
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
        self.database_dir = Path("vloc_workspace")
        self.match_database_dir = self.database_dir / "db.db"  # dir to database containing matches between queries and database
        self.workspace_original_database_dir = self.database_dir / "ori_database_hloc.db"

        self.all_queries_folder = "Test line"
        self.workspace_dir = Path('vloc_workspace_retrieval')
        self.workspace_images_dir = f"{self.workspace_dir}/images"
        self.workspace_loc_dir = f"{self.workspace_dir}/new"

        self.workspace_database_dir = self.workspace_dir / "database_hloc.db"
        self.workspace_queries_dir = f"{self.workspace_dir}/query"
        self.workspace_existing_model = f"{self.workspace_dir}/sparse"
        self.workspace_sfm_images_dir = f"{self.workspace_existing_model}/images.txt"
        self.workspace_sfm_point_cloud_dir = f"{self.workspace_existing_model}/points3D.txt"

        if self.workspace_database_dir.exists():
            print("Precomputed database found")
        else:
            self.create_hloc_database()

        self.image_to_kp_tree = {}
        self.point3d_cloud = None
        self.image_name_to_image_id = {}
        self.image2pose = None
        self.image2pose_new = None
        self.pid2features = None
        self.localization_results = []
        self.build_tree()
        self.id2kp_sfm, self.id2desc_sfm, self.id2name_sfm = colmap_db_read.extract_colmap_sift(self.match_database_dir)

        # matching database
        self.matches = None
        self.id2kp, self.id2desc, self.id2name, self.id2score = None, None, None, None
        self.h5_file_features = None

        self.desc_tree = None
        self.pid2images = None
        self.point3did2xyzrgb = colmap_io.read_points3D_coordinates(self.workspace_sfm_point_cloud_dir)
        self.point_cloud = None
        self.default_metadata = {'f': 26.0, 'cx': 1134.0, 'cy': 2016.0}

        # image retrieval variables
        self.retrieval_dataset = Path('vloc_workspace_retrieval')  # change this if your dataset is somewhere else
        self.retrieval_images_dir = self.retrieval_dataset / 'images_retrieval'
        self.retrieval_loc_pairs_dir = self.retrieval_dataset / 'pairs.txt'  # top 20 retrieved by NetVLAD
        self.retrieval_conf = extract_features.confs['netvlad']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_class = dynamic_load(extractors, self.retrieval_conf['model']['name'])
        self.retrieval_model = model_class(self.retrieval_conf['model']).eval().to(self.device)
        self.feature_path = Path(self.retrieval_dataset, self.retrieval_conf['output'] + '.h5')
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

        # matching variables
        self.image_ids = get_image_ids(self.workspace_database_dir)
        self.desc_type = None
        self.kp_type = None
        self.skip_geometric_verification = skip_geometric_verification
        self.matching_conf = extract_features.confs['sift']
        model_class = dynamic_load(extractors, self.matching_conf['model']['name'])
        self.matching_model = model_class(self.matching_conf['model']).eval().to(self.device)
        self.matching_feature_path = Path(self.retrieval_dataset, self.matching_conf['output'] + '.h5')
        self.matching_feature_path.parent.mkdir(exist_ok=True, parents=True)
        self.matching_skip_names = set(list_h5_names(self.matching_feature_path)
                                       if self.matching_feature_path.exists() else ())
        self.name2ref = match_features_bare.return_name2ref(self.matching_feature_path,
                                                            matches=self.retrieval_dataset / "matches.h5")

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
        self.query_names = pairs_from_retrieval.parse_names("query", None, query_names_h5)
        self.db_desc = pairs_from_retrieval.get_descriptors(self.db_names, db_descriptors, name2db)
        self.db_desc = self.db_desc.to(self.device)
        self.build_desc_tree()

        return

    def build_tree(self):
        self.image2pose = colmap_io.read_images(self.workspace_sfm_images_dir)
        self.pid2features = colmap_io.read_pid2images(self.image2pose)
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

    def create_folders(self):
        process = subprocess.Popen(["cp", self.workspace_original_database_dir, self.workspace_database_dir])
        return process

    def create_hloc_database(self):
        """
        Run this when hloc database needs to be reset
        """
        create_empty_db(self.workspace_original_database_dir)
        import_images(self.retrieval_images_dir, self.workspace_original_database_dir, "PER_FOLDER", None)
        extract_features.main_wo_model_loading(self.matching_model, self.device, [],
                                               self.matching_conf, self.retrieval_images_dir,
                                               feature_path=self.matching_feature_path)
        image_ids = get_image_ids(self.workspace_original_database_dir)
        database_image_ids = {u: v for u, v in image_ids.items() if "db" in u}
        import_features(database_image_ids, self.workspace_original_database_dir, self.matching_feature_path)

    def run_image_retrieval_and_matching(self):
        # retrieve
        extract_retrieval_pairs(self.cnn_retrieval_net, self.state,
                                "vloc_workspace_retrieval/database_global_descriptors_0.pkl",
                                "vloc_workspace_retrieval/images_retrieval/query/query.jpg",
                                "vloc_workspace_retrieval/retrieval_pairs.txt",
                                multi_scale=False,
                                nb_neighbors=40)

        # match
        extract_features.main_wo_model_loading(self.matching_model, self.device, self.matching_skip_names,
                                               self.matching_conf, self.retrieval_images_dir,
                                               feature_path=self.matching_feature_path)
        matching_conf = match_features_bare.confs["NN-ratio"]
        match_features_bare.main(self.name2ref, matching_conf, self.retrieval_loc_pairs_dir, self.matching_feature_path,
                                 matches=self.retrieval_dataset / "matches.h5", overwrite=True)

    def read_matches(self):
        with open(self.retrieval_loc_pairs_dir, 'r') as f:
            pairs = [p.split() for p in f.readlines()]
        image_ids = get_image_ids(self.workspace_database_dir)
        self.matches = {}
        with h5py.File(str(self.retrieval_dataset / "matches.h5"), 'r') as hfile:
            for p1, p2 in pairs:
                m1, score = get_matches_wo_loading(hfile, p1, p2)
                key_ = image_ids[p1], image_ids[p2]
                self.matches[key_] = (m1, score)

    def terminate(self):
        if self.h5_file_features is not None:
            self.h5_file_features.close()
        self.h5_file_features = None

    def read_features(self):
        self.h5_file_features = h5py.File(self.matching_feature_path, 'r')

    def build_desc_tree(self):
        self.id2name = {}
        with h5py.File(self.matching_feature_path, 'r') as hfile:
            if self.desc_tree is None:
                self.desc_tree = {}
                for name in self.image_ids:
                    img_id = self.image_ids[name]
                    self.id2name[img_id] = name
                    if "query" not in name:
                        desc_mat = np.transpose(hfile[name]["descriptors"].__array__())
                        self.desc_tree[img_id] = KDTree(desc_mat)

    def get_kp(self, img_id, fid):
        name = self.id2name[img_id]
        return self.h5_file_features[name]["keypoints"][fid]

    def get_desc(self, img_id, fid):
        name = self.id2name[img_id]
        return self.h5_file_features[name]["descriptors"][:, fid]

    def main_for_api(self, metadata):
        name_list = [f for f in listdir(self.workspace_queries_dir) if isfile(join(self.workspace_queries_dir, f))]
        self.run_image_retrieval_and_matching()
        self.read_matching_database()

        pairs, _ = self.read_2d_2d_matches(f"query/{name_list[0]}", debug=DEBUG)
        if len(pairs) == 0:
            return 0, None

        best_score = None
        best_pose = None
        for _ in range(10):
            r_mat, t_vec, score = localize(metadata, pairs)
            if best_score is None or score > best_score:
                best_score = score
                best_pose = (r_mat, t_vec)
                if best_score > 0.9:
                    break
        r_mat, t_vec = best_pose

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

    def gather_matches(self, arr, id1, query_im_id, database_im_id, desc_heuristics):
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

            database_fid_coord = self.get_kp(database_im_id, database_fid)  # hloc
            query_fid_coord = self.get_kp(query_im_id, query_fid)  # hloc

            x1, y1 = map(int, query_fid_coord)
            x2, y2 = map(int, database_fid_coord)
            query_fid_desc = None
            database_fid_desc = None
            ratio_test = None
            if desc_heuristics:
                query_fid_desc = self.get_desc(database_im_id, database_fid)  # hloc
                database_fid_desc = self.get_desc(query_im_id, query_fid)  # hloc
                dis, ind_list = self.desc_tree[database_im_id].query(query_fid_desc, 2)
                ratio_test = dis[0] / dis[1]

            pair = ((x1, y1), (x2, y2), query_fid_desc, database_fid_desc, ratio_test)
            pairs.append(pair)
            points1.append(query_fid_coord)
            points2.append(database_fid_coord)
        return points1, points2, pairs

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
            query_fid_coord, database_fid_coord, query_fid_desc, database_fid_desc, ratio_test = pair
            if desc_heuristics:
                desc_diff = np.sqrt(np.sum(np.square(query_fid_desc - database_fid_desc))) / 128
            else:
                desc_diff = 1
            fid_list, pid_list, tree, _ = self.image_to_kp_tree[database_im_id_sfm]  # sfm
            dis, ind = tree.query(database_fid_coord, 1)  # sfm
            pid = pid_list[ind]
            fid = fid_list[ind]
            candidate = MatchCandidate(query_fid_coord, fid, pid, dis, desc_diff, ratio_test)
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
                arr, score_mat = self.matches[m]
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
                                                                  database_im_id, desc_heuristics)
                    points1 = np.vstack(points1)
                    points2 = np.vstack(points2)

                    # homography ransac loop checking
                    verified, mask = self.verify_matches(points1, points2, pairs, query_im_id, database_im_id)
                    if not verified:
                        nb_skipped += 1
                        continue

                    pairs = filter_pairs(pairs, mask)
                    using += 1
                    if DEBUG:
                        log_matching(pairs,
                                     f"{self.retrieval_images_dir}/{self.id2name[database_im_id]}",
                                     f"{self.retrieval_images_dir}/{self.id2name[query_im_id]}",
                                     f"debug2/{GLOBAL_COUNT}-{points1.shape[0]}-useful.jpg")

                    self.register_matches(pairs, database_im_id_sfm, point3d_candidate_pool, desc_heuristics)

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
                print(cand_idx, candidate.dis,
                      candidate.desc_diff, point3d_candidate_pool.pid2votes[candidate.pid],
                      candidate.ratio_test, candidate.ratio_test_old)
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
