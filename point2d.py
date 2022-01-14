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
        self.cid2prob = None  # prob to sample cluster
        self.cluster_centers_ = None
        self.cid2kp = None  # kp belongs to this cluster and prob to sample features from this cluster
        self.cid2fprob = None  # prob to sample a kp for each cluster

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

    def sort_by_feature_strength(self, idx=0):
        indices = list(range(len(self.points)))
        indices = sorted(indices, key=lambda du: self.points[du].strength[idx], reverse=True)
        return indices

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
        distances, indices = self.desc_tree.query(query_desc, 5)
        coord_first = self.points[indices[0]].xy
        chosen_idx = 1
        for idx in range(1, 5):
            diff = np.sqrt(np.sum(np.square(coord_first - self.points[indices[idx]].xy)))
            if diff > 10:
                chosen_idx = idx
                break

        if distances[chosen_idx] > 0.0:
            ratio = distances[0] / distances[chosen_idx]
            if ratio < 0.7:
                return indices[0], distances[0], ratio
            else:
                return None, distances[0], ratio
        return None, distances[0], 1.0

    def matching_3d_to_2d_brute_force_no_ratio_test(self, query_desc):
        """
        brute forcing match for a single 3D point w/o ratio test
        """
        if self.desc_tree is None:
            self.desc_tree = KDTree(self.point_desc_list)
        res = self.desc_tree.query(query_desc, 2)
        return res[1][0], res[0][0], res[0][0] / res[0][1]

    def compute_top_n_prob(self, cid2kp, nb_clusters):
        top_k = 10
        cid2res = {}
        for cid in cid2kp:
            kp_list = cid2kp[cid]
            feature_strengths = [self.points[kp].strength[0] for kp in kp_list]
            feature_strengths = sorted(feature_strengths, reverse=True)
            feature_strengths = feature_strengths[:top_k+1]
            cid2res[cid] = np.mean(feature_strengths)
        return cid2res

    def cluster(self, nb_clusters=5, debug=False):
        data = [[self.points[du].xy[0],
                 self.points[du].xy[1],
                 self.points[du].strength[0],
                 self.points[du].strength[1]] for du in range(len(self.points))]
        cluster_model = MiniBatchKMeans(nb_clusters, random_state=1)
        clusters = cluster_model.fit_predict(data)
        cid2res = {idx: cluster_model.cluster_centers_[idx, 2] for idx in range(nb_clusters)}
        cid2kp = {idx: [] for idx in range(nb_clusters)}
        for kp, cid in enumerate(clusters):
            cid2kp[cid].append(kp)
        cid2res = self.compute_top_n_prob(cid2kp, nb_clusters)
        cid2prob = sampling_utils.combine([
            sampling_utils.rank_by_response(cid2res)
        ])
        for cid in cid2kp:
            kp_list = cid2kp[cid]
            prob_list = np.array([self.points[kp].strength[0] for kp in kp_list])
            prob_list /= np.sum(prob_list)
            prob_list2 = np.array([self.points[kp].strength[1] for kp in kp_list])
            prob_list2 /= np.sum(prob_list2)
            cid2kp[cid] = (kp_list, (prob_list+prob_list2)*0.5)
        self.cid2prob, self.cluster_centers_, self.cid2kp = cid2prob, cluster_model.cluster_centers_, cid2kp

        if debug:
            cluster2color = {du3: np.random.random((3,)) * 255.0 for du3 in range(nb_clusters)}
            for c in cluster2color:
                if c not in [1, 2, 3]:
                    cluster2color[c] = (0, 0, 0)
            img = np.copy(self.image)
            cid2prob2 = cid2prob.copy()
            while len(cid2prob2) > 0:
                cid = max(list(cid2prob2.keys()), key=lambda du: cid2prob2[du])
                del cid2prob2[cid]
                for feature_ind in cid2kp[cid][0]:
                    x, y = list(map(int, self.points[feature_ind].xy))
                    cv2.circle(img, (x, y), 20, cluster2color[clusters[feature_ind]], -1)
            return img
        return cid2prob, cluster_model.cluster_centers_, cid2kp

    def sample_by_feature_strengths(self, point3d_cloud, top_k=5, nb_samples=100, visibility_filtering=True):
        feature_indices = list(range(len(self)))
        feature_indices = sorted(feature_indices, key=lambda du: self[du].strength)
        r_list = np.zeros((len(feature_indices),))
        database = []
        all_points = []
        while True:
            if len(database) >= nb_samples:
                break

            fid = feature_indices.pop()

            if r_list[fid] == 1:
                continue
            r_list[fid] = 1

            distances, indices = point3d_cloud.top_k_nearest_desc(self[fid].desc, top_k)
            nb_desc = np.sum([len(point3d_cloud[point_ind].multi_desc_list) for point_ind in indices])
            database.append((nb_desc, fid, indices, distances))
            all_points.extend(indices)

        if visibility_filtering:
            count = {}
            for pid in all_points:
                gid = point3d_cloud[pid].visibility_graph_index
                if gid not in count:
                    count[gid] = 1
                else:
                    count[gid] += 1
            max_count = max(count.values())
            filter_database = []
            for nb_desc, fid, indices, distances in database:
                failed = True
                for pid in indices:
                    gid = point3d_cloud[pid].visibility_graph_index
                    if count[gid] == max_count:
                        failed = False
                if not failed:
                    filter_database.append((nb_desc, fid, indices, distances))
            return filter_database
        return database

    def sample(self, point3d_cloud, top_k=5, nb_samples=1000):
        self.cluster(nb_clusters=5)
        fid2cid = {}
        for cid in self.cid2kp:
            fid_list, _ = self.cid2kp[cid]
            for fid in fid_list:
                fid2cid[fid] = cid
        feature_indices = list(range(len(self)))
        cluster_indices = list(range(len(self.cid2kp)))
        cluster_probabilities = np.ones((len(self.cid2kp),)) * 1 / len(self.cid2kp)
        cluster_probabilities_based_on_feature_strengths = np.ones((len(self.cid2kp),)) * 1 / len(
            self.cid2kp)
        for c in self.cid2prob:
            cluster_probabilities_based_on_feature_strengths[c] = self.cid2prob[c]

        feature_probabilities = np.array([1 / len(feature_indices) for _ in range(len(feature_indices))])
        nb_desc_list = np.zeros_like(cluster_probabilities)
        count_desc_list = np.zeros_like(cluster_probabilities)
        r_list = np.zeros_like(feature_probabilities)
        database = []

        # exploring
        for _ in range(2):
            for cid in cluster_indices:
                fid_list, _ = self.cid2kp[cid]
                fid = np.random.choice(fid_list)
                if r_list[fid] == 1:
                    continue
                r_list[fid] = 1
                distances, indices = point3d_cloud.top_k_nearest_desc(self[fid].desc, top_k)
                total_nb_desc = np.sum([len(point3d_cloud[point_ind].multi_desc_list) for point_ind in indices])
                nb_desc_list[cid] += total_nb_desc
                count_desc_list[cid] += 1
                database.append((total_nb_desc, fid, indices, distances))

        # exploiting
        while True:
            if len(database) >= nb_samples:
                break
            cluster_probabilities = np.zeros((len(cluster_indices),))
            a1 = nb_desc_list / count_desc_list
            non_zero_idx = np.nonzero(a1 > 2)
            if non_zero_idx[0].shape[0] > 0:
                base_prob = 1 / len(cluster_indices)
                cluster_probabilities[non_zero_idx] = a1[non_zero_idx] * base_prob
            prob_sum = np.sum(cluster_probabilities)

            if prob_sum <= 0.0:
                continue
            sampling_prob = 0.7 * cluster_probabilities_based_on_feature_strengths + 0.3 * cluster_probabilities / prob_sum

            cid = np.random.choice(cluster_indices, p=sampling_prob / np.sum(sampling_prob))
            fid_list, _ = self.cid2kp[cid]
            fid = np.random.choice(fid_list)

            if r_list[fid] == 1:
                continue
            r_list[fid] = 1

            distances, indices = point3d_cloud.top_k_nearest_desc(self[fid].desc, top_k)
            nb_desc = np.sum([len(point3d_cloud[point_ind].multi_desc_list) for point_ind in indices])
            database.append((nb_desc, fid, indices, distances))

        return database


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

