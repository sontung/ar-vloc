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

    def desc_nearest(self, desc, nb_neighbors=2):
        pass

    def add_point(self, desc, xy):
        a_point = Feature(desc, xy)
        self.points.append(a_point)
        self.point_xy_list.append(xy)
        self.point_desc_list.append(desc)

    def commit(self):
        self.desc_tree = KDTree(self.point_desc_list)

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

