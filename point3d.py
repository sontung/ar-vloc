from scipy.spatial import KDTree
from feature_matching import build_vocabulary_of_descriptors
import time
import numpy as np


class PointCloud:
    def __init__(self):
        self.points = []
        self.point_id_list = []
        self.point_desc_list = []
        self.point_xyz_list = []
        self.xyz_tree = None
        self.desc_tree = None
        self.vocab, self.cluster_model = None, None

    def add_point(self, index, desc, xyz, rgb):
        a_point = Point3D(index, desc, xyz, rgb)
        self.points.append(a_point)
        self.point_id_list.append(index)
        self.point_xyz_list.append(xyz)
        self.point_desc_list.append(desc)

    def commit(self):
        self.xyz_tree = KDTree(self.point_xyz_list)
        self.desc_tree = KDTree(self.point_desc_list)
        self.vocab, self.cluster_model = build_vocabulary_of_descriptors(self.point_id_list,
                                                                         self.point_desc_list)

    def xyz_nearest(self, xyz, nb_neighbors=5):
        res = self.xyz_tree.query(xyz, nb_neighbors)
        return res

    def desc_nearest(self, desc, nb_neighbors=2):
        res = self.desc_tree.query(desc, nb_neighbors)
        return res

    def matching_2d_to_3d_brute_force(self, query_desc):
        """
        brute forcing match for a single 2D point
        """
        if self.desc_tree is None:
            print("Descriptor tree not built, use matching_2d_to_3d_vocab_based instead")
            raise AttributeError
        res = self.desc_tree.query(query_desc, 2)
        if res[0][1] > 0.0:
            if res[0][0] / res[0][1] < 0.7:  # ratio test
                index = res[1][0]
                return self.points[index]
        return None

    def matching_2d_to_3d_vocab_based(self, query_desc_list):
        start_time = time.time()
        result = []

        # assign each desc to a word
        desc_list = np.array(query_desc_list)
        words = self.cluster_model.predict(desc_list)

        # sort feature by search cost
        features_to_match = [
            (
                du, desc_list[du],
                len(self.vocab[words[du]]),
                self.vocab[words[du]]
            )
            for du in range(desc_list.shape[0])
        ]
        features_to_match = sorted(features_to_match, key=lambda du: du[2])

        for j, desc, _, point_3d_list in features_to_match:
            qu_point_3d_desc_list = [du2[1] for du2 in point_3d_list]
            qu_point_3d_id_list = [du2[0] for du2 in point_3d_list]

            kd_tree = KDTree(qu_point_3d_desc_list)
            res = kd_tree.query(desc, 2)
            if res[0][1] > 0.0:
                if res[0][0] / res[0][1] < 0.7:  # ratio test
                    result.append([j, qu_point_3d_id_list[res[1][0]]])
            if len(result) >= 100:
                break
        time_spent = time.time() - start_time
        return result


class Point3D:
    def __init__(self, index, descriptor, xyz, rgb):
        self.index = index
        self.desc = descriptor
        self.xyz = xyz
        self.rgb = rgb
        self.visual_word = None

    def match(self, desc):
        pass

    def assign_visual_word(self):
        pass

