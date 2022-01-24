import sys
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from scipy.spatial import KDTree
from feature_matching import build_vocabulary_of_descriptors
from optimizer import exhaustive_search, run_qap
from vis_utils import visualize_matching_helper

import time
import kmeans1d
import cv2
import open3d as o3d
import numpy as np


def ratio_test(results):
    distances, indices = results
    first_ind = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != first_ind:
            if distances[i] > 0.0:
                ratio = distances[0] / distances[i]
                if ratio < 0.7:
                    return True, ratio
                else:
                    return False, ratio
            return False, 10


def enforce_consistency_ratio_test(database):
    matches = {}
    matches_reverse = {}
    new_database = []

    for p_id, f_id, dist, ratio in database:
        candidates = [(f_id, p_id, dist)]
        if ratio is not None:
            if f_id in matches:
                p_id2, dist2 = matches[f_id]
                pair2 = (f_id, p_id2, dist2)
                candidates.append(pair2)
                del matches[f_id]
                del matches_reverse[p_id2]

            if p_id in matches_reverse:
                f_id2, dist2 = matches_reverse[p_id]
                pair2 = (f_id2, p_id, dist2)
                candidates.append(pair2)
                del matches[f_id2]
                del matches_reverse[p_id]
            f_id, p_id, dist = min(candidates, key=lambda du: du[-1])
            matches[f_id] = (p_id, dist)
            matches_reverse[p_id] = (f_id, dist)
        else:
            if f_id not in matches and p_id not in matches_reverse:
                new_database.append((p_id, f_id, dist, 0))
    for p_id in matches_reverse:
        f_id, dist = matches_reverse[p_id]
        new_database.append((p_id, f_id, dist, 0))
    print(f"After enforcing uniqueness constraint using ratio test, "
          f"reducing from {len(database)} to {len(new_database)}")
    return new_database


def enforce_consistency_distance(database):
    matches = {}
    matches_reverse = {}
    for p_id, f_id, dist, ratio in database:
        candidates = [(f_id, p_id, dist)]

        if f_id in matches:
            p_id2, dist2 = matches[f_id]
            pair2 = (f_id, p_id2, dist2)
            candidates.append(pair2)
            del matches[f_id]
            del matches_reverse[p_id2]

        if p_id in matches_reverse:
            f_id2, dist2 = matches_reverse[p_id]
            pair2 = (f_id2, p_id, dist2)
            candidates.append(pair2)
            del matches[f_id2]
            del matches_reverse[p_id]
        f_id, p_id, dist = min(candidates, key=lambda du: du[-1])
        matches[f_id] = (p_id, dist)
        matches_reverse[p_id] = (f_id, dist)
    new_database = []
    for p_id in matches_reverse:
        f_id, dist = matches_reverse[p_id]
        new_database.append((p_id, f_id, dist, 0))
    print(f"After enforcing uniqueness constraint using distance, "
          f"reducing from {len(database)} to {len(new_database)}")
    return new_database


class PointCloud:
    def __init__(self, multiple_desc_map, debug=False):
        self.points = []
        self.point_id_list = []
        self.point_desc_list = []
        self.point_indices_for_desc_tree = []
        self.point_xyz_list = []
        self.id2point = {}
        self.visibility_map = {}
        self.visibility_graph = {}
        self.point2map = {}
        self.multiple_desc_map = multiple_desc_map  # point id => multiple descriptors
        self.xyz_tree = None
        self.desc_tree = None
        self.debug = debug
        self.vocab, self.cluster_model = None, None
        self.pose_cluster_to_points = {}
        self.pose_cluster_to_image = {}
        self.image2points = {}
        self.point2images = {}
        self.point2co_visible_points = {}
        self.visibility_trees = []
        self.visibility_matrix = None

    def cluster(self, image2pose):
        self.cluster_by_pose(image2pose)
        self.cluster_by_position()

    def cluster_by_pose(self, image2pose):
        # cluster by poses
        pose_arr = []
        for image_id in image2pose:
            pose = image2pose[image_id][2]
            pose_arr.append(pose)
        if len(pose_arr) < 10:
            nb_clusters = 5
        else:
            nb_clusters = 10
        cluster_model = MiniBatchKMeans(nb_clusters, random_state=1)
        pose_arr = np.vstack(pose_arr)
        labels = cluster_model.fit_predict(pose_arr)
        for du in range(nb_clusters):
            self.pose_cluster_to_points[du] = []
            self.pose_cluster_to_image[du] = []
        for image_ind, image in enumerate(list(image2pose.keys())):
            data = image2pose[image]
            cam_pose_ind = labels[image_ind]
            self.pose_cluster_to_image[cam_pose_ind].append(data)
            for x, y, pid in data[1]:
                if pid > 0 and pid in self.id2point:
                    index = self.id2point[pid]
                    self.pose_cluster_to_points[cam_pose_ind].append(index)
        print("Pose clustering results:")
        for cam_pose_ind in self.pose_cluster_to_image:
            print(f" {cam_pose_ind} {len(self.pose_cluster_to_points[cam_pose_ind])} {[du3[0] for du3 in self.pose_cluster_to_image[cam_pose_ind]]}")

    def cluster_by_position(self):
        # cluster by positions
        for cid in self.pose_cluster_to_points:
            position_arr = []
            index_arr = self.pose_cluster_to_points[cid]
            for pid in index_arr:
                position_arr.append(self.points[pid].xyzrgb)
            position_arr = np.vstack(position_arr)
            nb_clusters = 20
            position_cluster_to_points = {pos_cid: [] for pos_cid in range(nb_clusters)}
            cluster_model = MiniBatchKMeans(nb_clusters, random_state=1)
            labels = cluster_model.fit_predict(position_arr)
            for ind, pos_cid in enumerate(labels):
                position_cluster_to_points[pos_cid].append(index_arr[ind])
            for pos_cid in position_cluster_to_points:
                pid_list = position_cluster_to_points[pos_cid]
                pid_degree_list = np.array([len(self.points[pid].multi_desc_list) for pid in pid_list])
                pid_prob_arr = pid_degree_list.astype(np.float64)
                pid_prob_arr /= np.sum(pid_prob_arr)
                position_cluster_to_points[pos_cid] = (pid_list, pid_prob_arr)
            prob_arr = np.ones((len(position_cluster_to_points),))*1/len(position_cluster_to_points)
            self.pose_cluster_to_points[cid] = [index_arr, position_cluster_to_points, prob_arr]

    def build_visibility_map(self, image2pose):
        for map_id, image in enumerate(list(image2pose.keys())):
            data = image2pose[image]
            pid_list = []
            self.visibility_map[map_id] = []
            for x, y, pid in data[1]:
                if pid > 0 and pid in self.id2point:
                    index = self.id2point[pid]
                    pid_list.append(index)
                    self.points[index].assign_visibility(data[0], (x, y))
                    if index in self.point2map:
                        self.point2map[index].append(map_id)
                    else:
                        self.point2map[index] = [map_id]
            self.visibility_map[map_id] = pid_list
        for index in self.point2map:
            graph_id = tuple(self.point2map[index])
            self.points[index].visibility_graph_index = graph_id
            if graph_id not in self.visibility_graph:
                data = []
                for gid in graph_id:
                    data.extend(self.visibility_map[gid])
                data = list(set(data))
                xyz_list = [self.point_xyz_list[du] for du in data]
                self.visibility_graph[graph_id] = (KDTree(xyz_list), data)
        print(f"Done building visibility graph of size {len(self.visibility_graph)}")

    def build_visibility_matrix(self, image2pose):
        for pid in range(len(self.points)):
            self.point2images[pid] = []
        for map_id, image in enumerate(list(image2pose.keys())):
            data = image2pose[image]
            self.image2points[map_id] = []
            for x, y, pid in data[1]:
                if pid > 0 and pid in self.id2point:
                    index = self.id2point[pid]
                    self.image2points[map_id].append(index)
                    self.point2images[index].append(map_id)
        for pid in range(len(self.points)):
            if pid not in self.point2co_visible_points:
                pid_list = self.access_visibility_matrix(pid)
                xyz_list = [self.point_xyz_list[du] for du in pid_list]
                tree = KDTree(xyz_list)
                self.visibility_trees.append(tree)
                self.point2co_visible_points[pid] = (len(self.visibility_trees)-1, pid_list)
                for pid2 in pid_list:
                    self.point2co_visible_points[pid2] = (len(self.visibility_trees) - 1, pid_list)
        assert len(self.point2co_visible_points) == len(self.points)
        print(f"Done building {len(self.visibility_trees)} visibility trees.")

    def access_visibility_matrix(self, pid):
        images = self.point2images[pid]
        pid_list = []
        for im_id in images:
            pid_list.extend(self.image2points[im_id])
        return list(set(pid_list))

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        return self.points[item]

    def add_point(self, index, desc, desc_list, xyz, rgb):
        a_point = Point3D(index, desc, xyz, rgb)
        a_point.multi_desc_list = desc_list
        a_point.compute_differences_between_descriptors()
        self.points.append(a_point)
        self.point_id_list.append(index)
        self.point_xyz_list.append(xyz)
        self.point_desc_list.append(desc)
        assert index not in self.id2point
        self.id2point[index] = len(self.points)-1

    def access_by_id(self, pid):
        if pid not in self.id2point:
            return None
        return self.points[self.id2point[pid]]

    def build_desc_tree(self):
        assert len(self.point_indices_for_desc_tree) == 0
        desc_list = []
        for index, point in enumerate(self.points):
            for desc in point.multi_desc_list:
                desc_list.append(desc)
                self.point_indices_for_desc_tree.append(index)
        self.desc_tree = KDTree(desc_list)

    def commit(self, image2pose):
        self.build_visibility_map(image2pose)
        self.xyz_tree = KDTree(self.point_xyz_list)
        if self.debug:
            self.build_desc_tree()
        self.vocab, self.cluster_model = build_vocabulary_of_descriptors(self.point_id_list,
                                                                         self.point_desc_list)
        print("Point cloud committed")

    def xyz_nearest(self, xyz, nb_neighbors=5):
        _, indices = self.xyz_tree.query(xyz, nb_neighbors)
        return indices

    def xyz_nearest_and_covisible(self, point_index, nb_neighbors=5):
        tree_idx, pid_list = self.point2co_visible_points[point_index]
        distances, _ = self.visibility_trees[tree_idx].query(self.points[point_index].xyz, nb_neighbors)
        indices = self.visibility_trees[tree_idx].query_ball_point(self.points[point_index].xyz, distances[-1])
        return [pid_list[du] for du in indices]

    def top_k_nearest_desc(self, query_desc, nb_neighbors):
        if self.desc_tree is None:
            raise AttributeError("Descriptor tree not built, use matching_2d_to_3d_vocab_based instead")
        distances, indices = self.desc_tree.query(query_desc, nb_neighbors)
        p_indices = [self.point_indices_for_desc_tree[i] for i in indices]
        return distances, p_indices

    def matching_2d_to_3d_brute_force_no_ratio_test(self, query_desc):
        """
        brute forcing match for a single 2D point
        """
        if self.desc_tree is None:
            raise AttributeError("Descriptor tree not built, use matching_2d_to_3d_vocab_based instead")
        res = self.desc_tree.query(query_desc, 1)
        index = self.point_indices_for_desc_tree[res[1]]
        nb_neighbors = len(self.points[index].multi_desc_list) + 1
        res = self.desc_tree.query(query_desc, nb_neighbors)
        positive, ratio = ratio_test(res)
        return index, res[0][0], res[0][1], ratio

    def matching_2d_to_3d_brute_force(self, query_desc, returning_index=False):
        """
        brute forcing match for a single 2D point
        """
        if self.desc_tree is None:
            raise AttributeError("Descriptor tree not built, use matching_2d_to_3d_vocab_based instead")
        res = self.desc_tree.query(query_desc, 1)
        index = self.point_indices_for_desc_tree[res[1]]
        nb_neighbors = len(self.points[index].multi_desc_list)+1
        res = self.desc_tree.query(query_desc, nb_neighbors)
        positive, ratio = ratio_test(res)
        if positive:
            index = self.point_indices_for_desc_tree[res[1][0]]
            if returning_index:
                return index, res[0][0], res[0][1], ratio
            return self.points[index], res[0][0], res[0][1], ratio

        return None, res[0][0], res[0][1], ratio

    def matching_2d_to_3d_vocab_based(self, feature_cloud, debug=False):
        result = []
        count = 0
        samples = 0

        # assign each desc to a word
        query_desc_list = feature_cloud.point_desc_list
        desc_list = np.array(query_desc_list)
        words = self.cluster_model.predict(desc_list)

        # sort feature by search cost
        features_to_match = [
            (
                du,
                desc_list[du],
                len(self.vocab[words[du]]),
                self.vocab[words[du]]
            )
            for du in range(desc_list.shape[0])
        ]
        features_to_match = sorted(features_to_match, key=lambda du: du[2])

        for j, desc, _, point_3d_list in features_to_match:
            qu_point_3d_desc_list = [du2[1] for du2 in point_3d_list]
            qu_point_3d_id_list = [du2[2] for du2 in point_3d_list]

            kd_tree = KDTree(qu_point_3d_desc_list)
            res = kd_tree.query(desc, 2)
            if res[0][1] > 0.0:
                if res[0][0] / res[0][1] < 0.7:  # ratio test
                    ans = self.points[qu_point_3d_id_list[res[1][0]]]
                    result.append([feature_cloud.points[j], ans])
                    if debug:
                        ref_res = self.matching_2d_to_3d_brute_force(desc)
                        samples += 1
                        if ref_res is not None and ans == ref_res:
                            count += 1

            if len(result) >= 100:
                break
        return result, count, samples

    def compute_feature_difference(self, query_desc, pid):
        smallest_dis = None
        for desc in self.points[pid].multi_desc_list:
            dis = np.sqrt(np.sum(np.square(query_desc - desc)))
            if smallest_dis is None or dis < smallest_dis:
                smallest_dis = dis
        return smallest_dis

    def search_neighborhood(self, database, point2d_cloud):
        ori_len = len(database)
        ori_database = database[:]
        pid_neighbors = []
        fid_neighbors = []
        correct_pairs = []
        for pid, fid, dis, ratio in ori_database:
            pid_neighbors.extend(self.xyz_nearest_and_covisible(pid, nb_neighbors=10))
            fid_neighbors.extend(point2d_cloud.nearby_feature(fid, nb_neighbors=100))
            correct_pairs.append((pid, fid))

        # filter duplicate
        pid_neighbors = list(set(pid_neighbors))
        fid_neighbors = list(set(fid_neighbors))
        must_fid = [pair[1] for pair in correct_pairs]
        fid_neighbors = point2d_cloud.filter_duplicate_features(fid_neighbors, must_fid)
        new_correct_pairs = correct_pairs[:]
        for index, (pid, fid) in enumerate(correct_pairs):
            new_correct_pairs[index] = (pid_neighbors.index(pid), fid_neighbors.index(fid))

        pid_desc_list = np.vstack([self[pid2].desc for pid2 in pid_neighbors])
        fid_desc_list = np.vstack([point2d_cloud[fid2].desc for fid2 in fid_neighbors])
        pid_coord_list = np.vstack([self[pid2].xyz for pid2 in pid_neighbors])
        fid_coord_list = np.vstack([point2d_cloud[fid2].xy for fid2 in fid_neighbors])

        print(f"Solving smoothness for {len(pid_neighbors)} points and {len(fid_neighbors)} features")
        solution = run_qap(pid_neighbors, fid_neighbors, pid_desc_list,
                           fid_desc_list, pid_coord_list, fid_coord_list, new_correct_pairs)
        only_neighborhood_database = []
        for u, v in solution:
            dis = self.compute_feature_difference(point2d_cloud[v].desc, u)
            if (u, v) not in database:
                database.append((u, v, dis, None))
                only_neighborhood_database.append((u, v, dis, None))

        print(f"Neighborhood search gains {len(database)-ori_len} extra matches.")
        return database, only_neighborhood_database

    def sample(self, point2d_cloud, image_ori, debug=True, fixed_database=True):
        if fixed_database:
            database = [(4004, 9173, 0.12098117412262448, 0.4402885800224702), (4021, 9035, 0.18678693412288302, 0.606487034384174), (4523, 9173, 0.173769433002931, 0.5529564433997507), (4533, 9124, 0.18860495123409068, 0.6202539604834014), (3242, 9124, 0.18090676986048645, 0.5661925920214619), (4001, 9207, 0.16749420519371688, 0.6066296138357196), (4535, 9207, 0.15554088565304214, 0.5375631689539395)]
            database = [du for du in database if du[0] not in [4523, 4533]]
            vis = False
            if vis:
                for pid, fid, dis, ratio in database:
                    print(pid, fid)
                    image = np.copy(image_ori)
                    fx, fy = point2d_cloud[fid].xy

                    fx, fy = map(int, (fx, fy))
                    cv2.circle(image, (fx, fy), 50, (128, 128, 0), -1)
                    image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
                    cv2.imshow("image", image)
                    cv2.waitKey()
                    images = visualize_matching_helper(np.copy(image), point2d_cloud[fid],
                                                       self.points[pid], "sfm_ws_hblab/images")
                    cv2.imshow("t", images)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
            return database
        else:
            visited_arr = np.zeros((len(self.points),))
            database, pose_cluster_prob_arr = self.sample_explore(point2d_cloud, visited_arr)
            database = self.sample_exploit(pose_cluster_prob_arr, database, visited_arr, point2d_cloud)
            print(database)
        print("Solve neighborhood smoothness with", [du[:2] for du in database])

        database, only_neighborhood_database = self.search_neighborhood(database, point2d_cloud)
        # database = enforce_consistency_ratio_test(database)
        # database = enforce_consistency_distance(database)
        results = []
        for pid, fid, dis, ratio in database:
            results.append([point2d_cloud[fid], self.points[pid]])
        if debug:
            # points_3d_list = []
            # pid_match_list = [du3[0] for du3 in database]
            # for pid in range(len(self.points)):
            #     x, y, z = self.points[pid].xyz
            #     r, g, b = 0, 0, 0
            #     if pid in pid_match_list:
            #         r, g, b = 1, 0, 0
            #     points_3d_list.append([x, y, z, r, g, b])
            # points_3d_list = np.vstack(points_3d_list)
            # point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d_list[:, :3]))
            # point_cloud.colors = o3d.utility.Vector3dVector(points_3d_list[:, 3:])
            # point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
            # point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            #
            # image = np.copy(image_ori)
            # for pid, fid, dis, ratio in database:
            #     x, y = map(int, point2d_cloud[fid].xy)
            #     cv2.circle(image, (x, y), 40, (255, 0, 0), -1)
            # image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
            # cv2.imshow("t", image)
            #
            # vis = o3d.visualization.Visualizer()
            # vis.create_window(width=1920, height=1025)
            # vis.add_geometry(point_cloud)
            # vis.run()
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # vis.destroy_window()

            for count, (pid, fid, _, _) in enumerate(tqdm(only_neighborhood_database,
                                                          desc="Visualizing for debugging")):
                image = np.copy(image_ori)
                fx, fy = point2d_cloud[fid].xy

                fx, fy = map(int, (fx, fy))
                cv2.circle(image, (fx, fy), 50, (128, 128, 0), -1)
                image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
                images = visualize_matching_helper(np.copy(image), point2d_cloud[fid],
                                                   self.points[pid], "sfm_ws_hblab/images")
                cv2.imwrite(f"debug/im-{count}.png", images)
                # cv2.imshow("t", images)
                # cv2.waitKey()
                # cv2.destroyAllWindows()

        return results

    def sample_matching_helper(self, pid, point2d_cloud):
        register = None
        smallest_ratio = 2.0
        for desc in self.points[pid].multi_desc_list:
            fid, dis, ratio = point2d_cloud.matching_3d_to_2d_brute_force(desc)
            if ratio < smallest_ratio:
                smallest_ratio = ratio
            if fid is not None:
                register = (fid, dis, ratio)
                break
        return register, smallest_ratio

    def sample_explore(self, point2d_cloud, visited_arr, visits_per_region=5):
        database = []
        pose_cluster_prob_arr = np.ones((len(self.pose_cluster_to_points),))*1.0/len(self.pose_cluster_to_points)
        print("Start with", pose_cluster_prob_arr.tolist())
        for pose_cluster_id in tqdm(self.pose_cluster_to_points, desc="Exploring"):
            _, position_cluster_to_points, prob_arr = self.pose_cluster_to_points[pose_cluster_id]
            assert len(prob_arr) == len(position_cluster_to_points)
            for position_cluster_id in position_cluster_to_points:
                pid_list, pid_prob_arr = position_cluster_to_points[position_cluster_id]
                for _ in range(visits_per_region):
                    pid = np.random.choice(pid_list, p=pid_prob_arr)
                    if visited_arr[pid] == 0:
                        register, ratio_s = self.sample_matching_helper(pid, point2d_cloud)
                        visited_arr[pid] = 1

                        if register is not None:
                            prob_arr[position_cluster_id] += 0.01
                            pose_cluster_prob_arr[pose_cluster_id] += 0.01
                            fid, dis, ratio = register
                            assert ratio == ratio_s
                            database.append((pid, fid, dis, ratio))

            self.pose_cluster_to_points[pose_cluster_id][2] = prob_arr/np.sum(prob_arr)
        pose_cluster_prob_arr /= np.sum(pose_cluster_prob_arr)
        print("After exploring", pose_cluster_prob_arr.tolist())
        print("Found", len(database))
        return database, pose_cluster_prob_arr

    def sample_exploit(self, pose_cluster_prob_arr, database, visited_arr, point2d_cloud):
        pose_cluster_list = list(range(len(self.pose_cluster_to_points)))
        counter = {pose_id: 0 for pose_id in pose_cluster_list}
        # while np.sum(pose_cluster_prob_arr) > 0:
        for _ in tqdm(range(5000), desc="Exploiting"):
            if len(database) > 500:
                break
            pose_cluster_id = np.random.choice(pose_cluster_list, p=pose_cluster_prob_arr)
            counter[pose_cluster_id] += 1
            _, position_cluster_to_points, prob_arr = self.pose_cluster_to_points[pose_cluster_id]
            position_cluster_id = np.random.choice(list(range(len(position_cluster_to_points))), p=prob_arr)
            pid_list, pid_prob_arr = position_cluster_to_points[position_cluster_id]
            pid_idx = np.random.choice(list(range(len(pid_list))), p=pid_prob_arr)
            pid = pid_list[pid_idx]
            if visited_arr[pid] == 0:
                register, ratio_s = self.sample_matching_helper(pid, point2d_cloud)
                visited_arr[pid] = 1

                if register is not None:
                    prob_arr[position_cluster_id] += 0.01
                    pose_cluster_prob_arr[pose_cluster_id] += 0.01
                    fid, dis, ratio = register
                    assert ratio == ratio_s
                    database.append((pid, fid, dis, ratio))

            # normalize
            pid_prob_arr[pid_idx] = 0
            if np.sum(pid_prob_arr) > 0:
                pid_prob_arr /= np.sum(pid_prob_arr)
            else:
                prob_arr[position_cluster_id] = 0
            position_cluster_to_points[position_cluster_id] = pid_list, pid_prob_arr
            self.pose_cluster_to_points[pose_cluster_id][1] = position_cluster_to_points
            if np.sum(prob_arr) > 0:
                self.pose_cluster_to_points[pose_cluster_id][2] = prob_arr / np.sum(prob_arr)
            else:
                pose_cluster_prob_arr[pose_cluster_id] = 0
            pose_cluster_prob_arr /= np.sum(pose_cluster_prob_arr)
        print(f"Found {len(database)} matches, after considering {int(np.sum(visited_arr))}.")
        print("After exploit", pose_cluster_prob_arr.tolist())
        print("Sampling times", counter)
        du2 = np.argmax(pose_cluster_prob_arr)
        print(pose_cluster_prob_arr[du2], du2, [du3[0] for du3 in self.pose_cluster_to_image[du2]])
        for idx, du in enumerate(pose_cluster_prob_arr):
            if du > 0.1:
                print(idx, du, [du3[0] for du3 in self.pose_cluster_to_image[idx]])
        print(f"Mean ratio={np.mean([du[-1] for du in database])}")
        return database


class Point3D:
    def __init__(self, index, descriptor, xyz, rgb):
        self.index = index
        self.multi_desc_list = []
        self.desc = descriptor
        self.xyz = xyz
        self.rgb = rgb
        self.xyzrgb = [xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2]]
        self.visual_word = None
        self.visibility = {}
        self.visibility_graph_index = None
        self.max_diff = None
        self.min_diff = None

    def compute_differences_between_descriptors(self):
        diff_list = []
        for d1 in self.multi_desc_list:
            for d2 in self.multi_desc_list:
                diff_list.append(np.sum(np.square(d1-d2)))
        self.max_diff = max(diff_list)
        self.min_diff = min(diff_list)

    def assign_visibility(self, im_name, coord):
        self.visibility[im_name] = coord

    def match(self, desc):
        pass

    def assign_visual_word(self):
        pass

    def __eq__(self, other):
        return self.index == other.index

