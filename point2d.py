import sys

from scipy.spatial import KDTree
from feature_matching import build_vocabulary_of_descriptors
from sklearn.cluster import MiniBatchKMeans
import sampling_utils
import time
import heapq
import numpy as np
import cv2
import kmeans1d


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
        self.cid2prob = None
        self.cluster_centers_ = None
        self.cid2kp = None
        self.cid_list = None
        self.cid_prob = None

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

    def rank_feature_strengths(self):
        ranks = [(idx, self.points[idx].strength) for idx in range(len(self.points))]
        ranks = sorted(ranks, key=lambda du: du[1], reverse=False)
        for rank, (idx, _) in enumerate(ranks):
            self.points[idx].strength_rank = (rank+1)/len(ranks)

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

    def cluster(self, nb_clusters=5, debug=False):
        data = [[self.points[du].xy[0],
                 self.points[du].xy[1],
                 self.points[du].strength] for du in range(len(self.points))]
        cluster_model = MiniBatchKMeans(nb_clusters, random_state=1)
        clusters = cluster_model.fit_predict(data)
        cid2res = {idx: cluster_model.cluster_centers_[idx, 2] for idx in range(nb_clusters)}
        cid2kp = {idx: [] for idx in range(nb_clusters)}
        for kp, cid in enumerate(clusters):
            cid2kp[cid].append(kp)
        cid2prob = sampling_utils.combine([
            sampling_utils.rank_by_area(cid2kp),
            sampling_utils.rank_by_response(cid2res)
        ])
        for cid in cid2kp:
            kp_list = cid2kp[cid]
            prob_list = np.array([self.points[kp].strength for kp in kp_list])
            prob_list /= np.sum(prob_list)
            cid2kp[cid] = (kp_list, prob_list)
        self.cid2kp = cid2kp
        if debug:
            cluster2color = {du3: np.random.random((3,)) * 255.0 for du3 in range(nb_clusters)}
            img = np.copy(self.image)
            cid2prob2 = cid2prob.copy()
            while len(cid2prob2) > 0:
                cid = max(list(cid2prob2.keys()), key=lambda du: cid2prob2[du])
                del cid2prob2[cid]
                for feature_ind in cid2kp[cid][0]:
                    x, y = list(map(int, self.points[feature_ind].xy))
                    cv2.circle(img, (x, y), 5, cluster2color[clusters[feature_ind]], -1)
            return img
        return cid2prob, cluster_model.cluster_centers_, cid2kp

    def sample(self):
        if self.cid2prob is None:
            self.cid2prob, self.cluster_centers_, self.cid2kp = self.cluster(debug=False)
            self.cid_list = [cid for cid in self.cid2prob.keys() if len(self.cid2kp[cid][0]) > 0]
            self.cid_prob = [self.cid2prob[cid] for cid in self.cid_list]

        try:
            random_cid = np.random.choice(self.cid_list, p=self.cid_prob)
        except ValueError:
            random_cid = np.random.choice(self.cid_list)

        fid_list, fid_prob_list = self.cid2kp[random_cid]
        random_fid = np.random.choice(fid_list)

        return random_fid


class Feature:
    def __init__(self, index, descriptor, xy, strength):
        self.desc = descriptor
        self.xy = xy
        self.visual_word = None
        self.index = index
        self.strength = strength
        self.strength_rank = -1

    def match(self, desc):
        pass

    def assign_visual_word(self):
        pass

