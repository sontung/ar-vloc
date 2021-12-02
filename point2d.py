from scipy.spatial import KDTree
from feature_matching import build_vocabulary_of_descriptors
import time
import numpy as np


class FeatureCloud:
    def __init__(self):
        self.points = []
        self.point_desc_list = []
        self.point_xy_list = []
        self.desc_tree = None
        self.vocab, self.cluster_model = None, None

    def desc_nearest(self, desc, nb_neighbors=2):
        pass

    def add_point(self, desc, xy):
        a_point = Feature(desc, xy)
        self.points.append(a_point)
        self.point_xy_list.append(xy)
        self.point_desc_list.append(desc)

    def commit(self):
        self.vocab, self.cluster_model = build_vocabulary_of_descriptors(list(range(len(self.points))),
                                                                         self.point_desc_list,
                                                                         nb_clusters=len(self.points) // 5)
        print("Feature cloud committed")

    def assign_search_cost(self, query_desc):
        query_desc = query_desc.reshape((1, -1))
        word = self.cluster_model.predict(query_desc)[0]
        return self.vocab[word]

    def matching_3d_to_2d_vocab_based(self, query_desc):
        word = self.cluster_model.predict(query_desc.reshape((1, -1)))[0]
        candidates = self.vocab[word]
        desc_list = [candidate[1] for candidate in candidates]
        id_list = [candidate[0] for candidate in candidates]
        tree = KDTree(desc_list)
        res = tree.query(query_desc, 2)
        if res[0][1] > 0.0:
            if res[0][0] / res[0][1] < 0.7:  # ratio test
                index = id_list[res[1][0]]
                return self.points[index]
        return None

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
    def __init__(self, descriptor, xy):
        self.desc = descriptor
        self.xy = xy
        self.visual_word = None

    def match(self, desc):
        pass

    def assign_visual_word(self):
        pass

