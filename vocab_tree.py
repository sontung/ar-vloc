import sys
import heapq
import numpy as np
import tqdm
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import KDTree
from vis_utils import visualize_matching

class VocabNode:
    def __init__(self, nb_clusters):
        self.nb_clusters = nb_clusters
        self.vocab = None
        self.cluster_model = None

    def __len__(self):
        return len(self.vocab)

    def assign_words(self, desc):
        return self.cluster_model.predict(desc)

    def traverse(self, word_id):
        return self.vocab[word_id]

    def build(self, point_cloud):
        p3d_id_list, p3d_desc_list = [], []
        for p3d_id in point_cloud.multiple_desc_map:
            for _, p3d_desc in point_cloud.multiple_desc_map[p3d_id]:
                p3d_id_list.append(p3d_id)
                p3d_desc_list.append(p3d_desc)
        if self.nb_clusters > len(p3d_id_list):
            raise AttributeError
        self.vocab = {u: [] for u in range(self.nb_clusters)}
        p3d_desc_list = np.array(p3d_desc_list)
        self.cluster_model = MiniBatchKMeans(self.nb_clusters, random_state=1)
        labels = self.cluster_model.fit_predict(p3d_desc_list)
        for ind in range(len(p3d_id_list)):
            point_cloud[point_cloud.id2point[p3d_id_list[ind]]].visual_word = labels[ind]
            self.vocab[labels[ind]].append((p3d_id_list[ind], p3d_desc_list[ind],
                                            point_cloud.id2point[p3d_id_list[ind]]))
        cluster_centers = self.cluster_model.cluster_centers_
        words = [Word(du, cluster_centers[du]) for du in range(cluster_centers.shape[0])]
        stats = [len(self.vocab[du]) for du in self.vocab]
        print(f"Visual words for 3D points:\n"
              f" max points per word={sorted(set(stats), reverse=True)[:5]}\n"
              f" min points per word={sorted(set(stats))[:5]}\n"
              f" mean points per word={np.mean(stats)}")
        return words


class Word:
    def __init__(self, index, descriptor):
        self.index = index
        self.descriptor = descriptor


class VocabNodeForWords:
    def __init__(self, name, level, branching_factor):
        self.vocab = None
        self.cluster_model = None
        self.branching_factor = branching_factor
        self.nb_clusters = branching_factor
        self.children = []
        self.name = name
        self.level = level

    def build(self, word_indices, word_descriptors):
        next_level = self.level + 1
        if self.level > 3:
            return []
        self.vocab = {u: [] for u in range(self.nb_clusters)}
        self.cluster_model = MiniBatchKMeans(self.nb_clusters, random_state=1)
        word_descriptors = np.vstack(word_descriptors)
        labels = self.cluster_model.fit_predict(word_descriptors)
        for i in range(len(word_indices)):
            self.vocab[labels[i]].append((word_indices[i], word_descriptors[i]))

        data = []
        for child_id in range(self.nb_clusters):
            child = VocabNodeForWords(f"{self.name}-{child_id}", next_level, self.branching_factor)
            sub_word_indices = [word[0] for word in self.vocab[child_id]]
            sub_word_descriptors = [word[1] for word in self.vocab[child_id]]
            data.extend([(word_ind, self.level, child.name, next_level) for word_ind in sub_word_indices])
            sub_data = child.build(sub_word_indices, sub_word_descriptors)
            data.extend(sub_data)
            self.children.append(child)
        return data


class VocabTree:
    def __init__(self, point_cloud, branching_factor=2):
        self.point_cloud = point_cloud
        self.v1 = VocabNode(nb_clusters=len(point_cloud)//50)
        words = self.v1.build(point_cloud)
        print(f"Constructing {len(self.v1)} visual words.")

        word_indices = [word.index for word in words]
        word_descriptors = [word.descriptor for word in words]

        self.v2 = VocabNodeForWords("0", 1, branching_factor)
        data = self.v2.build(word_indices, word_descriptors)
        self.matches = {}
        self.matches_reverse = {}
        self.word2level = {word_ind: {} for word_ind in word_indices}
        self.level2words = {}
        level_statistics = {du: 0 for du in range(4)}
        for word_ind, level, level_name, next_level in data:
            assert level not in self.word2level[word_ind]
            self.word2level[word_ind][level] = level_name
            if level_name not in self.level2words:
                self.level2words[level_name] = [word_ind]
            else:
                self.level2words[level_name].append(word_ind)
        for level_name in self.level2words:
            level_statistics[len(level_name.split("-"))-1] += 1
        print("Level statistics:", level_statistics)

    def enforce_consistency(self, pair, debug=False):
        candidates = [pair]
        f_id, p_id, dist = pair
        if f_id in self.matches:
            p_id2, dist2 = self.matches[f_id]
            pair2 = (f_id, p_id2, dist2)
            candidates.append(pair2)
            del self.matches[f_id]
            del self.matches_reverse[p_id2]

        if p_id in self.matches_reverse:
            f_id2, dist2 = self.matches_reverse[p_id]
            pair2 = (f_id2, p_id, dist2)
            candidates.append(pair2)
            del self.matches[f_id2]
            del self.matches_reverse[p_id]
        f_id, p_id, dist = min(candidates, key=lambda du: du[-1])
        self.matches[f_id] = (p_id, dist)
        self.matches_reverse[p_id] = (f_id, dist)

    def search_experimental(self, features, query_image_ori, sfm_image_folder, nb_matches=100, debug=False):
        self.matches.clear()
        self.matches_reverse.clear()

        count = 0
        samples = 0
        result = []

        # assign each desc to a word
        query_desc_list = features.point_desc_list
        query_desc_list = np.array(query_desc_list)
        words = self.v1.assign_words(query_desc_list)

        # sort feature by search cost
        if debug:
            feature_indices = [75044, 76121, 49563]
            for feature_ind in feature_indices:
                desc = features[feature_ind].desc

                ref_res, dist, _ = self.point_cloud.matching_2d_to_3d_brute_force(desc, returning_index=True)

                if ref_res is not None:
                    print(dist)
                    pair = (feature_ind, ref_res, dist)
                    self.enforce_consistency(pair)
                    visualize_matching([(features[feature_ind], None, dist)],
                                       [(features[feature_ind], self.point_cloud[ref_res], dist)],
                                       query_image_ori, sfm_image_folder)
                else:
                    print("not found", feature_ind)
        else:
            features_to_match = [
                (
                    -features[du].strength,
                    # len(self.v1.traverse(words[du])),
                    du,
                    query_desc_list[du],
                    self.v1.traverse(words[du]),
                    "feature"
                )
                for du in range(query_desc_list.shape[0])
                if len(self.v1.traverse(words[du])) > 20
            ]
            heapq.heapify(features_to_match)
            count = 0
            while len(self.matches) < nb_matches and len(features_to_match) > 0:
                candidate = heapq.heappop(features_to_match)
                # print(len(features_to_match), len(self.matches))
                if candidate[-1] == "feature":
                    count += 1
                    cost, feature_ind, desc, point_3d_list, _ = candidate
                    ref_res, dist, _ = self.point_cloud.matching_2d_to_3d_brute_force(desc, returning_index=True)
                    if ref_res is None:
                        continue
                    additional = self.nearby_check(feature_ind, ref_res, features)
                    if len(additional) > 0:
                        self.enforce_consistency((feature_ind, ref_res, dist))
                        # visualize_matching([(features[feature_ind], None, dist)],
                        #                    [(features[feature_ind], self.point_cloud[ref_res], dist)],
                        #                    query_image_ori, sfm_image_folder)
                        for pair in additional:
                            self.enforce_consistency(pair)
                            # feature_ind, ref_res, dist = pair
                            # visualize_matching([(features[feature_ind], None, dist)],
                            #                    [(features[feature_ind], self.point_cloud[ref_res], dist)],
                            #                    query_image_ori, sfm_image_folder)
            for f_id in self.matches:
                p_id, dist = self.matches[f_id]
                result.append((features[f_id], self.point_cloud[p_id], dist))
            print(f"Found {len(self.matches)} 2D-3D pairs, {len(features_to_match)} pairs left to consider.")
        return result

    def nearby_check(self, feature_ind, point_ind, features):
        nearby_points = list(self.point_cloud.xyz_nearest(self.point_cloud[point_ind].xyz))
        # nearby_points.append(point_ind)
        res = []
        for new_feature_ind in features.nearby_feature(feature_ind):
            pid, d, _ = self.point_cloud.matching_2d_to_3d_brute_force(features[new_feature_ind].desc,
                                                                       returning_index=True)
            if pid in nearby_points:
                res.append((new_feature_ind, pid, d))
        return res

    def search_brute_force(self, features, nb_matches=100, debug=False):
        self.matches.clear()
        self.matches_reverse.clear()

        count = 0
        samples = 0

        # assign each desc to a word
        query_desc_list = features.point_desc_list
        query_desc_list = np.array(query_desc_list)
        words = self.v1.assign_words(query_desc_list)

        # sort feature by search cost
        features_to_match = [
            (
                len(self.v1.traverse(words[du])),
                du,
                query_desc_list[du],
                self.v1.traverse(words[du]),
                "feature"
            )
            for du in range(query_desc_list.shape[0])
            if len(self.v1.traverse(words[du])) > 20
        ]
        heapq.heapify(features_to_match)

        while len(self.matches) < nb_matches and len(features_to_match) > 0:
            candidate = heapq.heappop(features_to_match)
            if candidate[-1] == "feature":
                _, feature_ind, desc, point_3d_list, _ = candidate
                ref_res, dist, _ = self.point_cloud.matching_2d_to_3d_brute_force(desc, returning_index=True)
                if ref_res is not None:
                    pair = (feature_ind, ref_res, dist)
                    self.enforce_consistency(pair)

        result = []
        for f_id in self.matches:
            p_id, dist = self.matches[f_id]
            result.append((features[f_id], self.point_cloud[p_id], dist))
        print(f"Found {len(self.matches)} 2D-3D pairs, {len(features_to_match)} pairs left to consider.")
        return result, count, samples

    def search(self, features, nb_matches=100, debug=False):
        self.matches.clear()
        self.matches_reverse.clear()

        count = 0
        samples = 0

        # assign each desc to a word
        query_desc_list = features.point_desc_list
        query_desc_list = np.array(query_desc_list)
        words = self.v1.assign_words(query_desc_list)

        # sort feature by search cost
        features_to_match = [
            (
                len(self.v1.traverse(words[du])),
                du,
                query_desc_list[du],
                self.v1.traverse(words[du]),
                "feature"
            )
            for du in range(query_desc_list.shape[0])
            if len(self.v1.traverse(words[du])) > 20
        ]
        heapq.heapify(features_to_match)

        while len(self.matches) < nb_matches and len(features_to_match) > 0:
            candidate = heapq.heappop(features_to_match)
            if candidate[-1] == "feature":
                _, feature_ind, desc, point_3d_list, _ = candidate
                qu_point_3d_desc_list = [du2[1] for du2 in point_3d_list]
                qu_point_3d_id_list = [du2[2] for du2 in point_3d_list]

                kd_tree = KDTree(qu_point_3d_desc_list)
                res_ = kd_tree.query(desc, 2)
                if res_[0][1] > 0.0:
                    if res_[0][0] / res_[0][1] < 0.7:  # ratio test
                        pair = (feature_ind, qu_point_3d_id_list[res_[1][0]], res_[0][0])
                        self.enforce_consistency(pair)

        # check consistency
        # for f_id in self.matches:
        #     p_id, dist = self.matches[f_id]
        #     f_id2, dist = self.matches_reverse[p_id]
        #     assert f_id2 == f_id

        result = []
        for f_id in self.matches:
            p_id, dist = self.matches[f_id]
            result.append((features[f_id], self.point_cloud[p_id], dist))
        print(f"Found {len(self.matches)} 2D-3D pairs, {len(features_to_match)} pairs left to consider.")

        if debug:
            bf_results = []
            for point_3d_index in self.matches_reverse:
                feature_ind, _ = self.matches_reverse[point_3d_index]
                desc_2d = features[feature_ind].desc
                ref_res, dist, _ = self.point_cloud.matching_2d_to_3d_brute_force(desc_2d)
                bf_results.append((features[feature_ind], ref_res, dist))
                samples += 1
                if ref_res is not None and self.point_cloud[point_3d_index] == ref_res:
                    count += 1
            return result, count, samples, bf_results

        return result, count, samples

    def active_search(self, features, nb_matches=100, debug=False):
        self.matches.clear()
        self.matches_reverse.clear()

        count = 0
        samples = 0

        if len(features) > 5000:
            level_to_use = 3
        else:
            level_to_use = 2

        # assign each desc to a word
        query_desc_list = features.point_desc_list
        query_desc_list = np.array(query_desc_list)
        words = self.v1.assign_words(query_desc_list)

        # sort feature by search cost
        features_to_match = [
            (
                len(self.v1.traverse(words[du])),
                du,
                query_desc_list[du],
                self.v1.traverse(words[du]),
                "feature"
            )
            for du in range(query_desc_list.shape[0])
            if len(self.v1.traverse(words[du])) > 20
        ]
        heapq.heapify(features_to_match)

        while len(self.matches) < nb_matches and len(features_to_match) > 0:
            candidate = heapq.heappop(features_to_match)
            if candidate[-1] == "feature":
                _, feature_ind, desc, point_3d_list, _ = candidate
                qu_point_3d_desc_list = [du2[1] for du2 in point_3d_list]
                qu_point_3d_id_list = [du2[2] for du2 in point_3d_list]

                kd_tree = KDTree(qu_point_3d_desc_list)
                res_ = kd_tree.query(desc, 2)
                if res_[0][1] > 0.0:
                    if res_[0][0] / res_[0][1] < 0.7:  # ratio test
                        ans = self.point_cloud[qu_point_3d_id_list[res_[1][0]]]
                        pair = (feature_ind, qu_point_3d_id_list[res_[1][0]], res_[0][0])
                        self.enforce_consistency(pair)

                        # re-check
                        # word_id = ans.visual_word
                        # level_name = self.word2level[word_id][level_to_use]
                        # new_candidate = (
                        #     0,
                        #     qu_point_3d_id_list[res_[1][0]],
                        #     level_name,
                        #     None,
                        #     "point"
                        # )
                        # heapq.heappush(features_to_match, new_candidate)

                        # potential checks
                        neighbors = self.point_cloud.xyz_nearest(ans.xyz)
                        for point_3d_index in neighbors:
                            word_id = self.point_cloud[point_3d_index].visual_word
                            level_name = self.word2level[word_id][level_to_use]
                            cost = features.assign_search_cost(level_name)
                            new_candidate = (
                                0,
                                point_3d_index,
                                level_name,
                                None,
                                "point"
                            )
                            heapq.heappush(features_to_match, new_candidate)
            elif candidate[-1] == "point":
                _, point_3d_index, level_name, _, _ = candidate
                res_ = features.search(level_name, self.point_cloud[point_3d_index].desc)

                if res_ is not None:
                    feature_ind, dist, desc_2d = res_
                    pair = (feature_ind, point_3d_index, dist)
                    self.enforce_consistency(pair)

                else:
                    if point_3d_index in self.matches_reverse:
                        fid_in_database, _ = self.matches_reverse[point_3d_index]
                        del self.matches[fid_in_database]
                        del self.matches_reverse[point_3d_index]

        if debug:
            for point_3d_index in self.matches_reverse:
                feature_ind, _ = self.matches_reverse[point_3d_index]
                desc_2d = features[feature_ind].desc
                ref_res, _, _ = self.point_cloud.matching_2d_to_3d_brute_force(desc_2d)
                samples += 1
                if ref_res is not None and self.point_cloud[point_3d_index] == ref_res:
                    count += 1

        result = []
        for f_id in self.matches:
            p_id, _ = self.matches[f_id]
            result.append((features[f_id], self.point_cloud[p_id]))
        print(f"Found {len(self.matches)} 2D-3D pairs, {len(features_to_match)} pairs left to consider.")
        return result, count, samples
