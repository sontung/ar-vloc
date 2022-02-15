import glob
import os
import pathlib
import subprocess
import sys

import cv2
import numpy as np
import colmap_db_read
import colmap_io
import colmap_read
import feature_matching
import localization
import shutil
import point2d
import point3d
import open3d as o3d
from vis_utils import visualize_matching_pairs, visualize_matching_helper, produce_cam_mesh, \
    visualize_cam_pose_with_point_cloud
from utils import to_homo
from scipy.spatial import KDTree
from distutils.dir_util import copy_tree
from tqdm import tqdm


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


def localize(metadata, pairs):
    f = metadata["f"] * 100
    cx = metadata["cx"]
    cy = metadata["cy"]

    r_mat, t_vec = localization.localize_single_image_lt_pnp(pairs, f, cx, cy)
    return r_mat, t_vec


def find_im_name(image2pose, name):
    for k in image2pose:
        res = image2pose[k]
        if res[0] == name:
            return res
    return None


class Localization:
    def __init__(self):
        self.database_dir = "colmap_loc"
        self.match_database_dir = "colmap_loc/database.db"  # dir to database containing matches between queries and database
        self.db_images_dir = "colmap_loc/images"
        self.query_images_folder = "/home/sontung/work/ar-vloc/colmap_loc/new_images"
        self.sfm_images_folder = "/home/sontung/work/ar-vloc/colmap_loc/images"
        self.ground_truth_dir = "/home/sontung/work/recon_models/office/images.txt"
        self.existing_model = "colmap_loc/sparse/0"

        self.all_queries_folder = "Test line"
        self.workspace_dir = "vloc_workspace"
        self.workspace_images_dir = f"{self.workspace_dir}/images"
        self.workspace_loc_dir = f"{self.workspace_dir}/new"
        self.workspace_sfm_point_cloud_dir = f"{self.workspace_dir}/new/points3D.txt"
        self.workspace_sfm_images_dir = f"{self.workspace_dir}/new/images.txt"
        self.workspace_database_dir = f"{self.workspace_dir}/database.db"
        self.workspace_existing_model = f"{self.workspace_dir}/sparse/0"

        self.image_to_kp_tree = {}
        self.point3d_cloud = None
        self.image_name_to_image_id = {}
        self.image2pose = None

        self.name_list, self.md_list, self.im_list = feature_matching.load_2d_queries_minimal(self.all_queries_folder)
        self.point3did2xyzrgb = None
        self.point_cloud = None
        return

    def build_point3d_cloud(self):
        point3d_id_list, point3d_desc_list, p3d_desc_list_multiple, point3did2descs = feature_matching.build_descriptors_2d_using_colmap_sift_no_verbose(
            self.image2pose, self.workspace_database_dir)
        self.point3d_cloud = point3d.PointCloud(point3did2descs, 0, 0, 0)

        for i in range(len(point3d_id_list)):
            point3d_id = point3d_id_list[i]
            point3d_desc = point3d_desc_list[i]
            xyzrgb = self.point3did2xyzrgb[point3d_id]
            self.point3d_cloud.add_point(point3d_id, point3d_desc, p3d_desc_list_multiple[i], xyzrgb[:3], xyzrgb[3:])

        self.point3d_cloud.commit(self.image2pose)

    def build_tree(self):
        self.image2pose = colmap_io.read_images(self.workspace_sfm_images_dir)
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
        self.create_folders()
        localization_results = []
        for idx in tqdm(range(len(self.name_list)), desc="Processing"):
            query_im = self.im_list[idx]
            cv2.imwrite(f"{self.workspace_images_dir}/query.jpg", query_im)

            tqdm.write(" run colmap")
            process = subprocess.Popen(["./colmap_match.sh"], shell=True, stdout=subprocess.PIPE)
            process.wait()
            tqdm.write(" done")

            self.point3did2xyzrgb = colmap_io.read_points3D_coordinates(self.workspace_sfm_point_cloud_dir)
            self.prepare_visualization()
            self.build_tree()
            self.build_point3d_cloud()
            pairs = self.match("query.jpg")
            metadata = self.md_list[idx]
            r_mat, t_vec = localize(metadata, pairs)
            localization_results.append(((r_mat, t_vec), (0, 1, 0)))

            # gt
            res = find_im_name(self.image2pose, "query.jpg")
            if res is not None:
                _, _, cam_pose, _ = res
                qw, qx, qy, qz, tx, ty, tz = cam_pose
                ref_rot_mat = colmap_read.qvec2rotmat([qw, qx, qy, qz])
                t_vec = np.array([tx, ty, tz])
                t_vec = t_vec.reshape((-1, 1))
                localization_results.append(((ref_rot_mat, t_vec), (0, 0, 0)))

            self.delete_folders()
        visualize_cam_pose_with_point_cloud(self.point_cloud, localization_results)

    def read_2d_2d_matches(self, query_im_name, only_geom_verified=False, threshold=5):
        matches, geom_matrices = colmap_db_read.extract_colmap_two_view_geometries(self.match_database_dir)
        id2kp, _, id2name = colmap_db_read.extract_colmap_sift(self.match_database_dir)
        name2id = {name: ind for ind, name in id2name.items()}
        query_im_id = name2id[query_im_name]
        pid_to_vote = {}
        pid_to_info = {}
        query_fid_to_database_fid = {}
        for m in matches:
            if query_im_id in m:
                arr = matches[m]
                conf, (f_mat, e_mat, h_mat) = geom_matrices[m]

                if arr is not None and f_mat is not None:
                    id1, id2 = m
                    if id1 != query_im_id:
                        database_im_id = id1
                    else:
                        database_im_id = id2
                    database_coord_list = []
                    kp_dict = {id1: [], id2: []}
                    print(f"checking pair {m} for query {query_im_id}")
                    for u, v in arr:
                        kp_dict[id1].append(u)
                        kp_dict[id2].append(v)
                        database_fid = kp_dict[database_im_id][-1]
                        query_fid = kp_dict[query_im_id][-1]

                        database_fid_coord = id2kp[database_im_id][database_fid]
                        fid_list, pid_list, tree, _ = self.image_to_kp_tree[database_im_id]
                        dis, ind = tree.query(database_fid_coord, 1)
                        if dis < threshold:
                            pid = pid_list[ind]
                            if query_fid not in query_fid_to_database_fid:
                                query_fid_to_database_fid[query_fid] = [(database_im_id, fid_list[ind], pid)]
                            else:
                                query_fid_to_database_fid[query_fid].append((database_im_id, fid_list[ind], pid))

                        database_coord_list.append(id2kp[database_im_id][database_fid])

        query_image = cv2.imread(f"{self.query_images_folder}/{id2name[query_im_id]}")
        for query_fid in query_fid_to_database_fid:
            coord = id2kp[query_im_id][query_fid]
            x2, y2 = map(int, coord)
            cv2.circle(query_image, (x2, y2), 20, (128, 128, 0), 10)
        image = cv2.resize(query_image, (query_image.shape[1] // 4, query_image.shape[0] // 4))
        cv2.imshow("", image)
        cv2.waitKey()
        cv2.destroyAllWindows()

        pid_results = [pid for pid in pid_to_vote]
        pid_results = sorted(pid_results, key=lambda du: pid_to_vote[du], reverse=True)
        for pid in pid_results[:15]:
            print(pid, pid_to_vote[pid])
                        # coord1 = id2kp[id1][u]
                        # coord2 = id2kp[id2][v]
                        # coord1 = to_homo(coord1).astype(np.float64)
                        # coord2 = to_homo(coord2).astype(np.float64)
                        # dis = cv2.sampsonDistance(coord1, coord2, f_mat.astype(np.float64))
                        # if dis < 1 or not only_geom_verified:
                        #     pair = (list(map(int, id2kp[id1][u])), list(map(int, id2kp[id2][v])), dis)
                        #     pairs.append(pair)
                    # image1 = cv2.imread(f"{self.db_images_dir}/{id2name[id1]}")
                    # image2 = cv2.imread(f"{self.db_images_dir}/{id2name[id2]}")
                    # image = visualize_matching_pairs(image1, image2, pairs)
                    # cv2.imwrite(f"colmap_sift/matches/img-{id1}-{id2}.jpg", image)
                    # image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
                    # cv2.imshow("", image)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()

    def match(self, query_im_name, debug=False):
        query_im_id = self.image_name_to_image_id[query_im_name]
        fid_list, pid_list, tree, coord_list = self.image_to_kp_tree[query_im_id]
        results = []
        for ind, pid in enumerate(pid_list):
            coord = coord_list[ind]
            point = self.point3d_cloud.access_by_id(pid)
            if point is None:
                continue
            results.append([coord, point.xyz])
            if debug:
                query_image = cv2.imread(f"{self.query_images_folder}/{query_im_name}")
                x2, y2 = map(int, coord)
                cv2.circle(query_image, (x2, y2), 50, (128, 128, 0), -1)
                images = visualize_matching_helper(query_image, 0, self.point3d_cloud.access_by_id(pid), self.sfm_images_folder)
                cv2.imwrite(f"debug/im-{ind}.jpg", images)
        return results

    def prepare_visualization(self):
        points_3d_list = []

        for point3d_id in self.point3did2xyzrgb:
            x, y, z, r, g, b = self.point3did2xyzrgb[point3d_id]
            points_3d_list.append([x, y, z, r / 255, g / 255, b / 255])
        points_3d_list = np.vstack(points_3d_list)
        self.point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d_list[:, :3]))
        self.point_cloud.colors = o3d.utility.Vector3dVector(points_3d_list[:, 3:])
        self.point_cloud, _ = self.point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

    def visualize(self):
        self.prepare_visualization()
        pairs = self.match("query.jpg")

        im_name_list, metadata_list, image_list = feature_matching.load_2d_queries_minimal("Test line small")

        p2d2p3d = {}
        for i in range(len(im_name_list)):
            p2d2p3d[i] = []

        localization_results = []
        for im_idx in p2d2p3d:
            metadata = metadata_list[im_idx]
            r_mat, t_vec = localize(metadata, pairs)
            localization_results.append(((r_mat, t_vec), (0, 1, 0)))

        visualize_cam_pose_with_point_cloud(self.point_cloud, localization_results)

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
    system.main()
    # system.visualize()
    # system.read_2d_2d_matches("query.jpg")
    # system.test_colmap_loc("query.jpg",
    #                        res_dir="/home/sontung/work/ar-vloc/colmap_loc/new/images.txt",
    #                        db_dir="/home/sontung/work/ar-vloc/colmap_loc/database.db")