from scipy.spatial import KDTree
from feature_matching import build_vocabulary_of_descriptors
import time
import numpy as np


class FeatureCloud:
    def __init__(self):
        self.points = []
        self.point_desc_list = []
        self.point_xy_list = []
        self.point_indices = []
        self.desc_tree = None
        self.level2features = {}

    def __getitem__(self, item):
        return self.points[item]

    def __len__(self):
        return len(self.points)

    def add_point(self, index, desc, xy):
        a_point = Feature(index, desc, xy)
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
            if res_[0][0] / res_[0][1] < 0.6:
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
            print("Descriptor tree not built, use matching_3d_to_2d_vocab_based instead")
            raise AttributeError
        res = self.desc_tree.query(query_desc, 2)
        if res[0][1] > 0.0:
            if res[0][0] / res[0][1] < 0.7:  # ratio test
                index = res[1][0]
                return self.points[index]
        return None


class Feature:
    def __init__(self, index, descriptor, xy):
        self.desc = descriptor
        self.xy = xy
        self.visual_word = None
        self.index = index

    def match(self, desc):
        pass

    def assign_visual_word(self):
        pass

