import sys
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from scipy.spatial import KDTree
from feature_matching import build_vocabulary_of_descriptors
import time
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
            print(f" {cam_pose_ind} {[du3[0] for du3 in self.pose_cluster_to_image[cam_pose_ind]]}")

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
        tree, data = self.visibility_graph[self.points[point_index].visibility_graph_index]
        _, indices = tree.query(self.points[point_index].xyz, nb_neighbors)
        return [data[du] for du in indices]

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

    def sample(self, point2d_cloud, image, debug=False):
        visited_arr = np.zeros((len(self.points),))
        database, pose_cluster_prob_arr = self.sample_explore(point2d_cloud, visited_arr)
        database = self.sample_exploit(pose_cluster_prob_arr, database, visited_arr, point2d_cloud)
        results = []
        for pid, fid, dis, ratio in database:
            results.append([point2d_cloud[fid], self.points[pid]])
        if debug:
            points_3d_list = []
            pid_match_list = [du3[0] for du3 in database]
            for pid in range(len(self.points)):
                x, y, z = self.points[pid].xyz
                r, g, b = 0, 0, 0
                if pid in pid_match_list:
                    r, g, b = 1, 0, 0
                points_3d_list.append([x, y, z, r, g, b])
            points_3d_list = np.vstack(points_3d_list)
            point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d_list[:, :3]))
            point_cloud.colors = o3d.utility.Vector3dVector(points_3d_list[:, 3:])
            point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
            point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            image = np.copy(image)
            for pid, fid, dis, ratio in database:
                x, y = map(int, point2d_cloud[fid].xy)
                cv2.circle(image, (x, y), 40, (255, 0, 0), -1)
            image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
            cv2.imshow("t", image)

            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1920, height=1025)
            vis.add_geometry(point_cloud)
            vis.run()
            cv2.waitKey()
            cv2.destroyAllWindows()
            vis.destroy_window()

        xyz_array = np.zeros((len(database), 3))
        xy_array = np.zeros((len(database), 2))
        for ind, (pid, fid, dis, ratio) in enumerate(database):
            xyz_array[ind] = self.points[pid].xyz
            xy_array[ind] = point2d_cloud[fid].xy
        with open('debug/test_refine.npy', 'wb') as f:
            np.save(f, xyz_array)
            np.save(f, xy_array)

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

    def sample_explore(self, point2d_cloud, visited_arr, visits_per_region=1):
        database = []
        pose_cluster_prob_arr = np.ones((len(self.pose_cluster_to_points),))*1.0/len(self.pose_cluster_to_points)
        print("Start with", pose_cluster_prob_arr.tolist())
        for pose_cluster_id in self.pose_cluster_to_points:
            _, position_cluster_to_points, prob_arr = self.pose_cluster_to_points[pose_cluster_id]
            assert len(prob_arr) == len(position_cluster_to_points)
            for position_cluster_id in position_cluster_to_points:
                pid_list, pid_prob_arr = position_cluster_to_points[position_cluster_id]
                for _ in range(visits_per_region):
                    pid = np.random.choice(pid_list, p=pid_prob_arr)
                    if visited_arr[pid] == 0:
                        register, ratio_s = self.sample_matching_helper(pid, point2d_cloud)
                        prob_arr[position_cluster_id] += 1 - ratio_s
                        pose_cluster_prob_arr[pose_cluster_id] += 1 - ratio_s
                        visited_arr[pid] = 1

                        if register is not None:
                            fid, dis, ratio = register
                            assert ratio == ratio_s
                            database.append((pid, fid, dis, ratio))
            self.pose_cluster_to_points[pose_cluster_id][2] = prob_arr/np.sum(prob_arr)
        pose_cluster_prob_arr /= np.sum(pose_cluster_prob_arr)
        print("After exploring", pose_cluster_prob_arr.tolist())
        return database, pose_cluster_prob_arr

    def sample_exploit(self, pose_cluster_prob_arr, database, visited_arr, point2d_cloud):
        pose_cluster_list = list(range(len(self.pose_cluster_to_points)))
        for _ in tqdm(range(5000), desc="Exploiting"):
            if len(database) > 100:
                break
            pose_cluster_id = np.random.choice(pose_cluster_list, p=pose_cluster_prob_arr)
            _, position_cluster_to_points, prob_arr = self.pose_cluster_to_points[pose_cluster_id]
            position_cluster_id = np.random.choice(list(range(len(position_cluster_to_points))), p=prob_arr)
            pid_list, pid_prob_arr = position_cluster_to_points[position_cluster_id]
            pid_idx = np.random.choice(list(range(len(pid_list))), p=pid_prob_arr)
            pid = pid_list[pid_idx]
            if visited_arr[pid] == 0:
                register, ratio_s = self.sample_matching_helper(pid, point2d_cloud)
                visited_arr[pid] = 1

                if register is not None:
                    prob_arr[position_cluster_id] += (1 - ratio_s) / 10
                    pose_cluster_prob_arr[pose_cluster_id] += (1 - ratio_s) / 10
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
        du2 = np.argmax(pose_cluster_prob_arr)
        print(pose_cluster_prob_arr[du2], du2, [du3[0] for du3 in self.pose_cluster_to_image[du2]])
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

