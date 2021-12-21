import sys

from scipy.spatial import KDTree
from feature_matching import build_vocabulary_of_descriptors
import time
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

    # @profile
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


class Point3D:
    def __init__(self, index, descriptor, xyz, rgb):
        self.index = index
        self.multi_desc_list = []
        self.desc = descriptor
        self.xyz = xyz
        self.rgb = rgb
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

