import glob
import os
import pathlib
import pickle
import random
import subprocess
import sys

import cv2
import numpy as np
import colmap_db_read
import colmap_io
import feature_matching
import localization
import shutil
import point3d
import open3d as o3d
from vis_utils import visualize_cam_pose_with_point_cloud, return_cam_mesh_with_pose
from vis_utils import concat_images_different_sizes, visualize_matching_helper_with_pid2features
from scipy.spatial import KDTree
from distutils.dir_util import copy_tree
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from pathlib import Path
from retrieval_utils import CandidatePool, MatchCandidate


def localize(metadata, pairs):
    f = metadata["f"] * 100
    cx = metadata["cx"]
    cy = metadata["cy"]

    r_mat, t_vec, score = localization.localize_single_image_lt_pnp(pairs, f, cx, cy, with_inliers_percent=True)
    return r_mat, t_vec, score


class Localization:
    def __init__(self):
        self.database_dir = "colmap_loc"
        self.match_database_dir = "colmap_loc/database.db"  # dir to database containing matches between queries and database
        self.db_images_dir = "colmap_loc/images"
        self.sfm_images_folder = "/home/sontung/work/ar-vloc/colmap_loc/images"
        self.ground_truth_dir = "/home/sontung/work/recon_models/office/images.txt"
        self.existing_model = "colmap_loc/sparse/0"

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
        self.pid2features = None

        self.point3did2xyzrgb = colmap_io.read_points3D_coordinates(self.workspace_sfm_point_cloud_dir)
        self.build_tree()
        self.build_point3d_cloud()
        self.point_cloud = None
        self.default_metadata = {'f': 26.0, 'cx': 1134.0, 'cy': 2016.0}
        return

    def build_point3d_cloud(self):
        my_file = Path(f"data/point3d_cloud.pkl")
        if my_file.is_file():
            with open(f"data/point3d_cloud.pkl", 'rb') as handle:
                self.point3d_cloud = pickle.load(handle)
        else:
            point3d_id_list, point3d_desc_list, p3d_desc_list_multiple, point3did2descs = feature_matching.build_descriptors_2d_using_colmap_sift_no_verbose(
                self.image2pose, self.workspace_database_dir)
            self.point3d_cloud = point3d.PointCloud(point3did2descs, 0, 0, 0)

            for i in range(len(point3d_id_list)):
                point3d_id = point3d_id_list[i]
                point3d_desc = point3d_desc_list[i]
                xyzrgb = self.point3did2xyzrgb[point3d_id]
                self.point3d_cloud.add_point(point3d_id, point3d_desc, p3d_desc_list_multiple[i], xyzrgb[:3], xyzrgb[3:])

            self.point3d_cloud.commit(self.image2pose)

            with open(f"data/point3d_cloud.pkl", 'wb') as handle:
                pickle.dump(self.point3d_cloud, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        pathlib.Path(self.workspace_images_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.workspace_loc_dir).mkdir(parents=True, exist_ok=True)

        shutil.copyfile(self.match_database_dir, f"{self.workspace_dir}/database.db")
        shutil.copyfile(f"{self.database_dir}/test_images.txt", f"{self.workspace_dir}/test_images.txt")

        copy_tree(self.db_images_dir, self.workspace_images_dir)
        copy_tree(self.existing_model, self.workspace_existing_model)

    def delete_folders(self):
        os.remove(f"{self.workspace_images_dir}/query.jpg")
        os.remove(f"{self.workspace_dir}/database.db")
        shutil.copyfile(self.match_database_dir, f"{self.workspace_dir}/database.db")

    def main(self):
        name_list = [f for f in listdir(self.workspace_queries_dir) if isfile(join(self.workspace_queries_dir, f))]
        # name_list = ["line-28.jpg", "line-22.jpg", "line-28.jpg"]
        localization_results = []
        failed = 0
        colmap_failed = 0
        failed_images = []
        for idx in tqdm(range(len(name_list)), desc="Processing"):
            pairs = self.read_2d_3d_matches(name_list[idx])
            if not pairs:
                colmap_failed += 1
                tqdm.write(f" {name_list[idx]} failed to loc, try voting-based...")
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
        print(f"Colmap failed {colmap_failed}, voting-based recovered {colmap_failed-failed}, total = {failed}"
              f" images: {failed_images}")
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

        image2pose = colmap_io.read_images("/home/sontung/work/ar-vloc/vloc_workspace_retrieval/new/images.txt")
        query_im_id = None
        for img_id in image2pose:
            image_name, points2d_meaningful, cam_pose, cam_id = image2pose[img_id]
            if image_name == query_im_name:
                query_im_id = img_id
        if query_im_id is None:
            return pairs
        else:
            image_name, points2d_meaningful, cam_pose, cam_id = image2pose[query_im_id]

            for x, y, pid in points2d_meaningful:
                if pid > -1:
                    xy = (x, y)
                    xyz = self.point3did2xyzrgb[pid][:3]
                    pairs.append((xy, xyz))
        return pairs

    def read_2d_2d_matches(self, query_im_name, debug=False):
        matches, geom_matrices = colmap_db_read.extract_colmap_two_view_geometries(self.workspace_database_dir)
        id2kp, id2desc, id2name = colmap_db_read.extract_colmap_sift(self.workspace_database_dir)

        name2id = {name: ind for ind, name in id2name.items()}
        query_im_id = name2id[query_im_name]
        point3d_candidate_pool = CandidatePool()
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
                    for u, v in arr:
                        kp_dict[id1].append(u)
                        kp_dict[id2].append(v)
                        database_fid = kp_dict[database_im_id][-1]
                        query_fid = kp_dict[query_im_id][-1]

                        database_fid_coord = id2kp[database_im_id][database_fid]
                        query_fid_coord = id2kp[query_im_id][query_fid]
                        query_fid_desc = id2desc[query_im_id][query_fid]

                        fid_list, pid_list, tree, _ = self.image_to_kp_tree[database_im_id]
                        distances, indices = tree.query(database_fid_coord, 10)
                        for idx, dis in enumerate(distances):
                            ind = indices[idx]
                            pid = pid_list[ind]
                            fid = fid_list[ind]
                            database_fid_desc = id2desc[database_im_id][fid]
                            desc_diff = np.sqrt(np.sum(np.square(query_fid_desc-database_fid_desc)))/128
                            candidate = MatchCandidate(query_fid_coord, fid, pid, dis, desc_diff)
                            point3d_candidate_pool.add(candidate)

        if len(point3d_candidate_pool) > 0:
            point3d_candidate_pool.count_votes()
            point3d_candidate_pool.filter()
            point3d_candidate_pool.sort(by_votes=True)
        pairs = []
        pid2images = colmap_io.read_pid2images(self.image2pose)
        for cand_idx, candidate in enumerate(point3d_candidate_pool.pool[:100]):
            pid = candidate.pid
            xy = candidate.query_coord
            xyz = self.point3did2xyzrgb[pid][:3]
            pairs.append((xy, xyz))

            if debug:
                print(candidate.pid, candidate.dis, point3d_candidate_pool.pid2votes[candidate.pid])
                x2, y2 = map(int, xy)
                query_img = cv2.imread(f"{self.workspace_queries_dir}/{query_im_name}")
                cv2.circle(query_img, (x2, y2), 50, (128, 128, 0), 10)
                vis_img = visualize_matching_helper_with_pid2features(query_img, pid2images[pid], self.workspace_images_dir)
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

    def test_colmap_loc(self, query_im_name,
                        res_dir="colmap_loc/sparse/0/images.txt",
                        db_dir="colmap_loc/not_full_database.db"):
        image2pose = colmap_io.read_images(res_dir)
        id2kp, _, id2name = colmap_db_read.extract_colmap_sift(db_dir)
        name2id = {name: ind for ind, name in id2name.items()}
        query_im_id = name2id[query_im_name]
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose[query_im_id]
        fid_list = []
        pid_list = []
        f_coord_list = []
        for fid, (fx, fy, pid) in enumerate(points2d_meaningful):
            if pid >= 0:
                fid_list.append(fid)
                pid_list.append(pid)
                f_coord_list.append([fx, fy])

        query_image = cv2.imread(f"{self.db_images_dir}/{id2name[query_im_id]}")
        for coord in f_coord_list:
            x2, y2 = map(int, coord)
            cv2.circle(query_image, (x2, y2), 20, (128, 128, 0), 10)
        image = cv2.resize(query_image, (query_image.shape[1] // 4, query_image.shape[0] // 4))
        cv2.imshow("", image)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    system = Localization()
    # system.debug_2d_2d_matches("line-28.jpg")
    # system.read_2d_2d_matches("line-28.jpg", debug=True)

    # system.debug_3d_mode("line-28.jpg")
    system.main()
    # system.read_2d_3d_matches("line-20.jpg")
