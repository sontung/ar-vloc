import subprocess
import time
import pycolmap
import torch
import cv2
import numpy as np
import colmap_db_read
import colmap_io
import localization
import open3d as o3d
from vis_utils import (visualize_cam_pose_with_point_cloud, return_cam_mesh_with_pose, visualize_matching_pairs,
                       concat_images_different_sizes, visualize_matching_helper_with_pid2features)
from scipy.spatial import KDTree
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from retrieval_utils import CandidatePool, MatchCandidate

import sys
sys.path.append("Hierarchical-Localization")

from pathlib import Path
from hloc import extractors
from hloc import extract_features, pairs_from_retrieval, match_features_bare

from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names
from hloc.triangulation import (import_features, import_matches, geometric_verification)
from hloc.reconstruction import create_empty_db, import_images, get_image_ids


def localize(metadata, pairs):
    f = metadata["f"] * 100
    cx = metadata["cx"]
    cy = metadata["cy"]

    r_mat, t_vec, score = localization.localize_single_image_lt_pnp(pairs, f, cx, cy, with_inliers_percent=True)
    return r_mat, t_vec, score


def api_test():
    system = Localization()
    query_folder = "test_line_jpg"
    name_list = [f for f in listdir(query_folder) if isfile(join(query_folder, f))]
    query_folder_workspace1 = "vloc_workspace_retrieval/images"
    query_folder_workspace2 = "vloc_workspace_retrieval/images_retrieval/query"
    default_metadata = {'f': 26.0, 'cx': 1134.0, 'cy': 2016.0}

    start = time.time()
    for name in name_list:
        query_img = cv2.imread(f"{query_folder}/{name}")
        cv2.imwrite(f"{query_folder_workspace1}/query.jpg", query_img)
        cv2.imwrite(f"{query_folder_workspace2}/query.jpg", query_img)
        system.main_for_api(default_metadata)
        # break
    end = time.time()
    print(f"Done in {end - start} seconds, avg. {(end - start) / len(name_list)} s/image")
    system.visualize()


class Localization:
    def __init__(self):
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
        self.id2kp, self.id2desc, self.id2name = None, None, None
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

        # matching variables
        self.skip_geometric_verification = False
        self.matching_conf = extract_features.confs['sift']
        model_class = dynamic_load(extractors, self.matching_conf['model']['name'])
        self.matching_model = model_class(self.matching_conf['model']).eval().to(self.device)
        self.matching_feature_path = Path(self.retrieval_dataset, self.matching_conf['output'] + '.h5')
        self.matching_feature_path.parent.mkdir(exist_ok=True, parents=True)
        self.matching_skip_names = set(list_h5_names(self.matching_feature_path)
                                       if self.matching_feature_path.exists() else ())
        self.name2ref = match_features_bare.return_name2ref(self.matching_feature_path,
                                                            matches=self.retrieval_dataset / "matches.h5")
        # self.create_hloc_database()

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
        create_empty_db(self.workspace_original_database_dir)
        import_images(self.retrieval_images_dir, self.workspace_original_database_dir, "PER_FOLDER", None)
        extract_features.main_wo_model_loading(self.matching_model, self.device, self.matching_skip_names,
                                               self.matching_conf, self.retrieval_images_dir,
                                               feature_path=self.matching_feature_path)
        image_ids = get_image_ids(self.workspace_original_database_dir)
        database_image_ids = {u: v for u, v in image_ids.items() if "db" in u}
        import_features(database_image_ids, self.workspace_original_database_dir, self.matching_feature_path)

    @profile
    def run_image_retrieval_and_matching(self):
        copy_process = self.create_folders()
        # retrieve
        global_descriptors = extract_features.main_wo_model_loading(self.retrieval_model, self.device, self.skip_names,
                                                                    self.retrieval_conf, self.retrieval_images_dir,
                                                                    feature_path=self.feature_path)
        pairs_from_retrieval.main_wo_loading(global_descriptors, self.retrieval_loc_pairs_dir, 40,
                                             self.db_names, self.db_desc, self.query_names, self.device)

        # match
        extract_features.main_wo_model_loading(self.matching_model, self.device, self.matching_skip_names,
                                               self.matching_conf, self.retrieval_images_dir,
                                               feature_path=self.matching_feature_path)
        matching_conf = match_features_bare.confs["NN-ratio"]
        match_features_bare.main(self.name2ref, matching_conf, self.retrieval_loc_pairs_dir, self.matching_feature_path,
                                 matches=self.retrieval_dataset / "matches.h5", overwrite=True)
        copy_process.wait()
        copy_process.kill()

        # create_empty_db(self.workspace_database_dir)
        # import_images(self.retrieval_images_dir, self.workspace_database_dir, "PER_FOLDER", None)
        image_ids = get_image_ids(self.workspace_database_dir)
        query_image_ids = {u: v for u, v in image_ids.items() if "query" in u}

        import_features(query_image_ids, self.workspace_database_dir, self.matching_feature_path)
        import_matches(image_ids, self.workspace_database_dir, self.retrieval_loc_pairs_dir,
                       self.retrieval_dataset / "matches.h5", None, self.skip_geometric_verification)
        if not self.skip_geometric_verification:
            geometric_verification(self.workspace_database_dir, self.retrieval_loc_pairs_dir)

    @profile
    def main_for_api(self, metadata):
        name_list = [f for f in listdir(self.workspace_queries_dir) if isfile(join(self.workspace_queries_dir, f))]
        self.run_image_retrieval_and_matching()
        self.read_matching_database(self.workspace_database_dir)

        pairs, _ = self.read_2d_2d_matches(f"query/{name_list[0]}")
        if len(pairs) == 0:
            return

        best_score = None
        best_pose = None
        for _ in range(10):
            r_mat, t_vec, score = localize(metadata, pairs)
            if best_score is None or score > best_score:
                best_score = score
                best_pose = (r_mat, t_vec)
                if best_score == 1.0:
                    break
        r_mat, t_vec = best_pose

        self.localization_results.append(((r_mat, t_vec), (1, 0, 0)))

    def visualize(self):
        self.prepare_visualization()
        visualize_cam_pose_with_point_cloud(self.point_cloud, self.localization_results)
        self.localization_results.clear()

    def read_matching_database(self, database_dir):
        self.matches, _ = colmap_db_read.extract_colmap_two_view_geometries(database_dir)
        self.id2kp, self.id2desc, self.id2name = colmap_db_read.extract_colmap_hloc(database_dir)

    def read_2d_2d_matches(self, query_im_name, debug=False, max_pool_size=100):
        name2id = {name: ind for ind, name in self.id2name.items()}
        desc_heuristics = False
        if len(self.id2desc) > 0:
            desc_heuristics = True
        query_im_id = name2id[query_im_name]
        point3d_candidate_pool = CandidatePool()
        for m in self.matches:
            if query_im_id in m:
                arr = self.matches[m]
                if arr is not None:
                    id1, id2 = m
                    if id1 != query_im_id:
                        database_im_id = id1
                    else:
                        database_im_id = id2
                    kp_dict = {id1: [], id2: []}

                    database_im_id_sfm = self.image_name_to_image_id[self.id2name[database_im_id].split("/")[-1]]

                    pairs = []
                    for u, v in arr:
                        kp_dict[id1].append(u)
                        kp_dict[id2].append(v)
                        database_fid = kp_dict[database_im_id][-1]
                        query_fid = kp_dict[query_im_id][-1]

                        database_fid_coord = self.id2kp[database_im_id][database_fid]  # hloc
                        query_fid_coord = self.id2kp[query_im_id][query_fid]  # hloc
                        query_fid_desc = None

                        x1, y1 = map(int, query_fid_coord)
                        x2, y2 = map(int, database_fid_coord)
                        pair = ((x1, y1), (x2, y2))
                        pairs.append(pair)

                        if desc_heuristics:
                            query_fid_desc = self.id2desc[query_im_id][query_fid]  # hloc

                        fid_list, pid_list, tree, _ = self.image_to_kp_tree[database_im_id_sfm]  # sfm
                        distances, indices = tree.query(database_fid_coord, 10)  # sfm
                        for idx, dis in enumerate(distances):
                            if dis < 10:
                                ind = indices[idx]  # sfm
                                pid = pid_list[ind]
                                fid = fid_list[ind]
                                if desc_heuristics:
                                    database_fid_desc = self.id2desc_sfm[database_im_id_sfm][ind]  # sfm
                                    desc_diff = np.sqrt(np.sum(np.square(query_fid_desc-database_fid_desc)))/128
                                else:
                                    desc_diff = 1
                                candidate = MatchCandidate(query_fid_coord, fid, pid, dis, desc_diff)
                                point3d_candidate_pool.add(candidate)
                    # if debug:
                    #     db_img = cv2.imread(f"{self.retrieval_images_dir}/{self.id2name[database_im_id]}")
                    #     query_img = cv2.imread(f"{self.retrieval_images_dir}/{self.id2name[query_im_id]}")
                    #     img = visualize_matching_pairs(query_img, db_img, pairs)
                    #     img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
                    #     cv2.imshow("", img)
                    #     cv2.waitKey()
                    #     cv2.destroyAllWindows()
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

            if debug:
                self.pid2images = colmap_io.read_pid2images(self.image2pose)
                print(candidate.pid, candidate.dis, point3d_candidate_pool.pid2votes[candidate.pid])
                x2, y2 = map(int, xy)
                query_img = cv2.imread(f"{self.retrieval_images_dir}/{query_im_name}")
                if query_img is None:
                    print(f"{self.retrieval_images_dir}/{query_im_name}")
                    raise ValueError
                cv2.circle(query_img, (x2, y2), 50, (128, 128, 0), 10)
                vis_img = visualize_matching_helper_with_pid2features(query_img, self.pid2images[pid],
                                                                      self.workspace_images_dir)
                cv2.imwrite(f"debug2/img-{cand_idx}.jpg", vis_img)
        tqdm.write(f" voting-based returns {len(pairs)} pairs for {query_im_name}")
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
