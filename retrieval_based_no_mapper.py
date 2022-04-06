import pathlib
import random
import subprocess
import time
import pycolmap
import torch
import cv2
import numpy as np
import colmap_db_read
import colmap_io
import colmap_read
import localization
import shutil
import open3d as o3d
from vis_utils import visualize_cam_pose_with_point_cloud, return_cam_mesh_with_pose
from vis_utils import concat_images_different_sizes, visualize_matching_helper_with_pid2features
from scipy.spatial import KDTree
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from retrieval_utils import CandidatePool, MatchCandidate
from utils import rewrite_retrieval_output
import os

import sys
sys.path.append("Hierarchical-Localization")

from pathlib import Path
from hloc import extractors
from hloc import extract_features, pairs_from_retrieval

from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names


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
    end = time.time()
    print(f"Done in {end - start} seconds, avg. {(end - start) / len(name_list)} s/image")
    system.visualize()


class Localization:
    def __init__(self):
        self.database_dir = "vloc_workspace"
        self.match_database_dir = f"{self.database_dir}/db.db"  # dir to database containing matches between queries and database

        self.all_queries_folder = "Test line"
        self.workspace_dir = "vloc_workspace_retrieval"
        self.workspace_images_dir = f"{self.workspace_dir}/images"
        self.workspace_loc_dir = f"{self.workspace_dir}/new"

        self.workspace_database_dir = f"{self.workspace_dir}/db.db"
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

        # matching database
        self.matches = None
        self.id2kp, self.id2desc, self.id2name = None, None, None
        self.pid2images = None

        self.point3did2xyzrgb = colmap_io.read_points3D_coordinates(self.workspace_sfm_point_cloud_dir)
        self.build_tree()
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
        pathlib.Path(self.workspace_loc_dir).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.match_database_dir, f"{self.workspace_dir}/db.db")

    def create_folders2(self):
        pathlib.Path(self.workspace_loc_dir).mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(["cp", self.match_database_dir, f"{self.workspace_dir}/db.db"])
        return process

    def delete_folders(self):
        os.remove(f"{self.workspace_dir}/db.db")

    def run_colmap(self, metadata):
        focal = metadata["f"] * 100
        cx = metadata["cx"]
        cy = metadata["cy"]
        os.chdir(self.workspace_dir)
        subprocess.run(["colmap", "feature_extractor",
                        "--database_path", "db.db",
                        "--image_path", "images",
                        "--image_list_path", "test_image.txt",
                        "--ImageReader.single_camera", "1",
                        "--ImageReader.camera_model", "PINHOLE",
                        "--ImageReader.camera_params", f"{focal}, {focal}, {cx}, {cy}"
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        # subprocess.run(["colmap", "matches_importer",
        #                 "--database_path", "db.db",
        #                 "--match_list_path", "retrieval_pairs.txt",
        #                 ])
        # subprocess.run(["colmap", "image_registrator",
        #                 "--database_path", "db.db",
        #                 "--input_path", "sparse",
        #                 "--output_path", "new",
        #                 "--Mapper.ba_refine_focal_length", "0",
        #                 "--Mapper.ba_refine_extra_params", "0",
        #                 ])
        subprocess.run(["sh", "colmap_match.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       stdin=subprocess.DEVNULL)
        os.chdir("..")

    def run_image_retrieval(self):
        global_descriptors = extract_features.main_wo_model_loading(self.retrieval_model, self.device, self.skip_names,
                                                                    self.retrieval_conf, self.retrieval_images_dir,
                                                                    feature_path=self.feature_path)
        pairs_from_retrieval.main_wo_loading(global_descriptors, self.retrieval_loc_pairs_dir, 40,
                                             self.db_names, self.db_desc, self.query_names, self.device)

    def main_for_api(self, metadata):
        copy_process = self.create_folders2()
        name_list = [f for f in listdir(self.workspace_queries_dir) if isfile(join(self.workspace_queries_dir, f))]

        tqdm.write(" run image retrieval")
        self.run_image_retrieval()
        tqdm.write(" done")

        rewrite_retrieval_output(f"{self.workspace_dir}/pairs.txt", f"{self.workspace_dir}/retrieval_pairs.txt")

        copy_process.wait()
        copy_process.kill()

        tqdm.write(" run colmap")
        self.run_colmap(metadata)
        tqdm.write(" done")

        loc_res = self.read_2d_3d_matches_pcm(name_list[0])
        if loc_res is None:
            self.read_matching_database()
            tqdm.write(f" {name_list[0]} failed to loc, try voting-based...")
            pairs, _ = self.read_2d_2d_matches(name_list[0])
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

            if best_score < 0.2:
                return
        else:
            r_mat, t_vec = loc_res
        self.localization_results.append(((r_mat, t_vec), (1, 0, 0)))

    def visualize(self):
        self.prepare_visualization()
        visualize_cam_pose_with_point_cloud(self.point_cloud, self.localization_results)
        self.localization_results.clear()

    def main(self, debug=False):
        start = time.time()
        self.create_folders()
        name_list = [f for f in listdir(self.workspace_queries_dir) if isfile(join(self.workspace_queries_dir, f))]
        localization_results = []
        failed = 0
        colmap_failed = 0
        failed_images = []

        self.run_commands()

        # reading colmap database
        self.image2pose_new = colmap_io.read_images(f"{self.workspace_loc_dir}/images.txt")
        self.read_matching_database()

        for idx in tqdm(range(len(name_list)), desc="Processing"):
            pairs = self.read_2d_3d_matches(name_list[idx])
            if not pairs:
                colmap_failed += 1
                tqdm.write(f" {name_list[idx]} failed to loc, try voting-based...")
                if debug:
                    self.verbose_when_failed(name_list[idx])
                pairs, _ = self.read_2d_2d_matches(name_list[idx])
                if len(pairs) == 0:
                    failed += 1
                    failed_images.append(name_list[idx])
                    continue

            best_score = None
            best_pose = None
            for _ in range(10):
                r_mat, t_vec, score = localize(self.default_metadata, pairs)
                if best_score is None or score > best_score:
                    best_score = score
                    best_pose = (r_mat, t_vec)
                    if best_score == 1.0:
                        break
            r_mat, t_vec = best_pose

            if best_score < 0.2:
                failed += 1
                failed_images.append(name_list[idx])
                continue
            localization_results.append(((r_mat, t_vec), (0, 1, 0)))
        self.delete_folders()

        end = time.time()
        print(f"Colmap failed {colmap_failed}, voting-based recovered {colmap_failed-failed}, total = {failed}"
              f" images: {failed_images}")
        print(f"Done in {end-start} seconds, avg. {(end-start)/len(name_list)} s/image")
        self.prepare_visualization()
        visualize_cam_pose_with_point_cloud(self.point_cloud, localization_results)

    def verbose_when_failed(self, query_im_name):
        matches_verified, _ = colmap_db_read.extract_colmap_two_view_geometries(self.workspace_database_dir)
        matches = colmap_db_read.extract_colmap_matches(self.workspace_database_dir)
        id2kp, _, id2name = colmap_db_read.extract_colmap_sift(self.workspace_database_dir)
        name2id = {name: ind for ind, name in id2name.items()}
        query_im_id = name2id[query_im_name]
        count = 0
        for m in matches_verified:
            if query_im_id in m:
                if matches_verified[m] is not None:
                    count += 1
        count2 = 0
        for m in matches:
            if query_im_id in m:
                if matches[m] is not None:
                    count2 += 1
        tqdm.write(f" {query_im_name} only matches {count}/{count2}")

    def debug_2d_2d_matches(self, query_im_name, write_to_file=True):
        matches, _ = colmap_db_read.extract_colmap_two_view_geometries(self.workspace_database_dir)
        id2kp, _, id2name = colmap_db_read.extract_colmap_sift(self.workspace_database_dir)
        name2id = {name: ind for ind, name in id2name.items()}
        query_im_id = name2id[query_im_name]
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
                    database_img = cv2.imread(f"{self.workspace_images_dir}/{id2name[database_im_id]}")
                    query_img = cv2.imread(f"{self.workspace_queries_dir}/{id2name[query_im_id]}")
                    vis_img = concat_images_different_sizes([database_img, query_img])

                    for u, v in arr:
                        kp_dict[id1].append(u)
                        kp_dict[id2].append(v)
                        database_fid = kp_dict[database_im_id][-1]
                        query_fid = kp_dict[query_im_id][-1]

                        database_fid_coord = id2kp[database_im_id][database_fid]
                        query_fid_coord = id2kp[query_im_id][query_fid]
                        x1, y1 = map(int, database_fid_coord)
                        x2, y2 = map(int, query_fid_coord)

                        color = (random.random()*255, random.random()*255, random.random()*255)
                        cv2.circle(vis_img, (x1, y1), 20, color, 5)
                        cv2.circle(vis_img, (x2+database_img.shape[1], y2), 20, color, 5)
                        cv2.line(vis_img, (x1, y1), (x2+database_img.shape[1], y2), color, 5)

                        # cv2.circle(database_img, (x1, y1), 20, (0, 255, 0), 5)
                        # cv2.circle(query_img, (x2, y2), 20, (255, 0, 0), 5)
                        # database_img = cv2.resize(database_img, (database_img.shape[1] // 4, database_img.shape[0] // 4))
                        # query_img = cv2.resize(query_img, (query_img.shape[1] // 4, query_img.shape[0] // 4))
                        # cv2.imshow("1", query_img)
                        # cv2.imshow("2", database_img)

                    if write_to_file:
                        cv2.imwrite(f"debug/img-{database_im_id}.jpg", vis_img)
                    else:
                        vis_img = cv2.resize(vis_img, (vis_img.shape[1] // 4, vis_img.shape[0] // 4))
                        cv2.imshow("", vis_img)
                        cv2.waitKey()
                        cv2.destroyAllWindows()

    def read_2d_3d_matches(self, query_im_name):
        pairs = []

        query_im_id = None
        for img_id in self.image2pose_new:
            image_name, points2d_meaningful, cam_pose, cam_id = self.image2pose_new[img_id]
            if image_name == query_im_name:
                query_im_id = img_id
        if query_im_id is None:
            return pairs
        else:
            image_name, points2d_meaningful, cam_pose, cam_id = self.image2pose_new[query_im_id]

            for x, y, pid in points2d_meaningful:
                if pid > -1:
                    xy = (x, y)
                    xyz = self.point3did2xyzrgb[pid][:3]
                    pairs.append((xy, xyz))
        return pairs

    def read_2d_3d_matches_pcm(self, query_im_name):
        reconstruction = pycolmap.Reconstruction(self.workspace_loc_dir)
        res = None
        for image_id, image in reconstruction.images.items():
            if image.name == query_im_name:
                qw, qx, qy, qz = image.qvec
                tx, ty, tz = image.tvec
                rot_mat = colmap_read.qvec2rotmat([qw, qx, qy, qz])
                t_vec = np.array([tx, ty, tz])
                res = (rot_mat, t_vec)
        return res

    def read_matching_database(self, database_dir=None):
        if database_dir is None:
            self.matches, _ = colmap_db_read.extract_colmap_two_view_geometries(self.workspace_database_dir)
            self.id2kp, self.id2desc, self.id2name = colmap_db_read.extract_colmap_sift(self.workspace_database_dir)
        else:
            self.matches, _ = colmap_db_read.extract_colmap_two_view_geometries(database_dir)
            self.id2kp, self.id2desc, self.id2name = colmap_db_read.extract_colmap_sift(database_dir)

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
                    for u, v in arr:
                        kp_dict[id1].append(u)
                        kp_dict[id2].append(v)
                        database_fid = kp_dict[database_im_id][-1]
                        query_fid = kp_dict[query_im_id][-1]

                        database_fid_coord = self.id2kp[database_im_id][database_fid]
                        query_fid_coord = self.id2kp[query_im_id][query_fid]
                        query_fid_desc = None

                        if desc_heuristics:
                            query_fid_desc = self.id2desc[query_im_id][query_fid]

                        fid_list, pid_list, tree, _ = self.image_to_kp_tree[database_im_id]
                        distances, indices = tree.query(database_fid_coord, 10)
                        for idx, dis in enumerate(distances):
                            ind = indices[idx]
                            pid = pid_list[ind]
                            fid = fid_list[ind]
                            if desc_heuristics:
                                print(self.id2desc[database_im_id].shape, fid)
                                database_fid_desc = self.id2desc[database_im_id][fid]
                                desc_diff = np.sqrt(np.sum(np.square(query_fid_desc-database_fid_desc)))/128
                            else:
                                desc_diff = 1
                            candidate = MatchCandidate(query_fid_coord, fid, pid, dis, desc_diff)
                            point3d_candidate_pool.add(candidate)

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
                query_img = cv2.imread(f"{self.workspace_queries_dir}/{query_im_name}")
                cv2.circle(query_img, (x2, y2), 50, (128, 128, 0), 10)
                vis_img = visualize_matching_helper_with_pid2features(query_img, self.pid2images[pid],
                                                                      self.workspace_images_dir)
                cv2.imwrite(f"debug2/img-{cand_idx}.jpg", vis_img)
        tqdm.write(f" voting-based returns {len(pairs)} pairs for {query_im_name}")
        return pairs, point3d_candidate_pool

    def debug_3d_mode(self, query_im_name):
        pairs, point3d_candidate_pool = self.read_2d_2d_matches(query_im_name)

        localization_results = []
        r_mat, t_vec, score = localize(self.default_metadata, pairs)
        localization_results.append(((r_mat, t_vec), (0, 1, 0)))
        ind = 0

        pid_list = []
        for candidate in point3d_candidate_pool.pool[:100]:
            ind += 1
            pid = candidate.pid
            pid_list.append(pid)

        points_3d_list = []
        for point3d_id in self.point3did2xyzrgb:
            x, y, z, r, g, b = self.point3did2xyzrgb[point3d_id]
            if point3d_id not in pid_list:
                points_3d_list.append([x, y, z, r / 255, g / 255, b / 255])
            else:
                points_3d_list.append([x, y, z, 1, 0, 0])

        points_3d_list = np.vstack(points_3d_list)
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d_list[:, :3]))
        point_cloud.colors = o3d.utility.Vector3dVector(points_3d_list[:, 3:])
        point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1025)
        vis.add_geometry(point_cloud)
        cameras = return_cam_mesh_with_pose(localization_results)
        for c in cameras:
            vis.add_geometry(c)
        vis.run()
        vis.destroy_window()
        return pairs

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

    # default_metadata = {'f': 26.0, 'cx': 1134.0, 'cy': 2016.0}
    # system.main_for_api2(default_metadata)
    # system.debug_2d_2d_matches("line-28.jpg")
    # system.read_2d_2d_matches("line-20.jpg", debug=True)

    # system.debug_3d_mode("line-28.jpg")
    # system.main()
    # system.prepare_visualization()
    # system.read_2d_3d_matches("line-20.jpg")
