import sys

from scipy.spatial import KDTree
from feature_matching import build_vocabulary_of_descriptors
import time
import heapq
import numpy as np
import cv2


class FeatureCloud:
    def __init__(self, image=None):
        self.image = image
        self.points = []
        self.point_desc_list = []
        self.point_xy_list = []
        self.point_indices = []
        self.desc_tree = None
        self.xy_tree = None
        self.level2features = {}

    def __getitem__(self, item):
        return self.points[item]

    def __len__(self):
        return len(self.points)

    def nearby_feature(self, ind, nb_neighbors=20, max_distance=50, min_distance=4,
                       strict_lower_bound=False, return_distances=False):
        if self.xy_tree is None:
            self.xy_tree = KDTree(self.point_xy_list)
        distances, indices = self.xy_tree.query(self.point_xy_list[ind], nb_neighbors,
                                                distance_upper_bound=max_distance)
        if nb_neighbors == 1:
            return [indices]
        res = []
        dis = []
        for i in range(len(distances)):
            if not strict_lower_bound:
                if min_distance <= distances[i] <= max_distance:
                    res.append(indices[i])
                    dis.append(distances[i])
            else:
                if min_distance < distances[i] <= max_distance:
                    res.append(indices[i])
                    dis.append(distances[i])
        if return_distances:
            return res, dis
        return res

    def add_point(self, index, desc, xy, response):
        a_point = Feature(index, desc, xy, response)
        self.points.append(a_point)
        self.point_indices.append(index)
        self.point_xy_list.append(xy)
        self.point_desc_list.append(desc)

    def search(self, level_name, query_desc):
        desc_list = [self.points[point_ind].desc for point_ind in self.level2features[level_name]]
        point_indices = [point_ind for point_ind in self.level2features[level_name]]
        tree = KDTree(desc_list)
        res_ = tree.query(query_desc, 2)
        if res_[0][1] > 0.0:
            if res_[0][0] / res_[0][1] < 0.7:
                desc_2d = self.point_desc_list[point_indices[res_[1][0]]]
                return point_indices[res_[1][0]], res_[0][0], desc_2d
        return None

    def assign_words(self, word2level, word_tree):
        self.level2features.clear()
        query_descriptors = np.vstack(self.point_desc_list)
        query_words = word_tree.assign_words(query_descriptors)
        for point_ind, word_id in enumerate(query_words):
            levels = list(word2level[word_id].values())
            for level_name in levels:
                if level_name not in self.level2features:
                    self.level2features[level_name] = [point_ind]
                else:
                    self.level2features[level_name].append(point_ind)

    def assign_search_cost(self, level_name):
        return len(self.level2features[level_name])

    def matching_3d_to_2d_brute_force(self, query_desc):
        """
        brute forcing match for a single 3D point
        """
        if self.desc_tree is None:
            self.desc_tree = KDTree(self.point_desc_list)
        res = self.desc_tree.query(query_desc, 2)
        if res[0][1] > 0.0:
            if res[0][0] / res[0][1] < 0.7:  # ratio test
                index = res[1][0]
                return index, res[0][0]
        return None, res[0][0]

    def matching_3d_to_2d_brute_force_no_ratio_test(self, query_desc):
        """
        brute forcing match for a single 3D point w/o ratio test
        """
        if self.desc_tree is None:
            self.desc_tree = KDTree(self.point_desc_list)
        res = self.desc_tree.query(query_desc, 2)
        return res[1][0], res[0][0], res[0][0] / res[0][1]

    def sample(self):
        features_to_match = [
            (
                -self.points[du].strength,
                du,
            )
            for du in range(len(self.points))
        ]
        heapq.heapify(features_to_match)
        img = np.copy(self.image)
        img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
        cv2.namedWindow('image')
        while len(features_to_match) > 0:
            candidate = heapq.heappop(features_to_match)
            _, feature_ind = candidate
            x, y = list(map(int, self.points[feature_ind].xy))
            cv2.circle(img, (x // 4, y // 4), 5, (255, 0, 0), 1)
            cv2.imshow("image", img)
            cv2.waitKey(1)
        sys.exit()
        return


class Feature:
    def __init__(self, index, descriptor, xy, strength):
        self.desc = descriptor
        self.xy = xy
        self.visual_word = None
        self.index = index
        self.strength = strength

    def match(self, desc):
        pass

    def assign_visual_word(self):
        pass

