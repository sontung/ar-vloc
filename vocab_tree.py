import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import KDTree


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
        p3d_id_list, p3d_desc_list = point_cloud.point_id_list, point_cloud.point_desc_list
        if self.nb_clusters > len(p3d_id_list):
            raise AttributeError
        self.vocab = {u: [] for u in range(self.nb_clusters)}
        p3d_desc_list = np.array(p3d_desc_list)
        self.cluster_model = MiniBatchKMeans(self.nb_clusters)
        labels = self.cluster_model.fit_predict(p3d_desc_list)
        for ind in range(len(p3d_id_list)):
            point_cloud[ind].visual_word = labels[ind]
            self.vocab[labels[ind]].append((p3d_id_list[ind], p3d_desc_list[ind], ind))
        cluster_centers = self.cluster_model.cluster_centers_
        words = [Word(du, cluster_centers[du]) for du in range(cluster_centers.shape[0])]
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
        self.cluster_model = MiniBatchKMeans(self.nb_clusters)
        word_descriptors = np.vstack(word_descriptors)
        labels = self.cluster_model.fit_predict(word_descriptors)
        for i in range(len(word_indices)):
            self.vocab[labels[i]].append((word_indices[i], word_descriptors[i]))

        data = []
        for child_id in range(self.nb_clusters):
            child = VocabNodeForWords(f"{self.name}-{child_id}", next_level, self.branching_factor)
            sub_word_indices = [word[0] for word in self.vocab[child_id]]
            sub_word_descriptors = [word[1] for word in self.vocab[child_id]]
            data.extend([(word_ind, self.level, child.name) for word_ind in sub_word_indices])
            sub_data = child.build(sub_word_indices, sub_word_descriptors)
            data.extend(sub_data)
            self.children.append(child)
        return data


class VocabTree:
    def __init__(self, point_cloud, branching_factor=3):
        self.point_cloud = point_cloud
        self.v1 = VocabNode(nb_clusters=len(point_cloud)//50)
        words = self.v1.build(point_cloud)
        print(f"Constructing {len(self.v1)} visual words.")

        word_indices = [word.index for word in words]
        word_descriptors = [word.descriptor for word in words]

        self.v2 = VocabNodeForWords("0", 1, branching_factor)
        data = self.v2.build(word_indices, word_descriptors)
        self.matches = {}
        self.word2level = {word_ind: {} for word_ind in word_indices}
        self.level2words = {}
        for word_ind, level, level_name in data:
            assert level not in self.word2level[word_ind]
            self.word2level[word_ind][level] = level_name
            if level_name not in self.level2words:
                self.level2words[level_name] = [word_ind]
            else:
                self.level2words[level_name].append(word_ind)

    def enforce_consistency(self, pair):
        f_id, p_id, dist = pair
        if f_id not in self.matches:
            self.matches[f_id] = (p_id, dist)
            return True
        else:
            _, prev_dist = self.matches[f_id]
            if dist < prev_dist:
                self.matches[f_id] = (p_id, dist)
                return True
        return False

    def search(self, features, debug=False):
        self.matches.clear()

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
                du,
                query_desc_list[du],
                self.v1.traverse(words[du]),
                len(self.v1.traverse(words[du])),
                "feature"
            )
            for du in range(query_desc_list.shape[0])
        ]

        while len(self.matches) < 100 and len(features_to_match) > 0:
            index = min(list(range(len(features_to_match))),
                        key=lambda du: features_to_match[du][-2])
            candidate = features_to_match[index]
            del features_to_match[index]
            if candidate[-1] == "feature":
                feature_ind, desc, point_3d_list, _, _ = candidate
                qu_point_3d_desc_list = [du2[1] for du2 in point_3d_list]
                qu_point_3d_id_list = [du2[2] for du2 in point_3d_list]

                kd_tree = KDTree(qu_point_3d_desc_list)
                res_ = kd_tree.query(desc, 2)
                if res_[0][1] > 0.0:
                    if res_[0][0] / res_[0][1] < 0.7:  # ratio test
                        ans = self.point_cloud[qu_point_3d_id_list[res_[1][0]]]
                        pair = (feature_ind, qu_point_3d_id_list[res_[1][0]], res_[0][0])
                        take_this = self.enforce_consistency(pair)

                        if debug and take_this:
                            ref_res = self.point_cloud.matching_2d_to_3d_brute_force(desc)
                            samples += 1
                            if ref_res is not None and ans == ref_res:
                                count += 1

                        neighbors = self.point_cloud.xyz_nearest(ans.xyz)
                        for point_3d_index in neighbors:
                            word_id = self.point_cloud[point_3d_index].visual_word
                            level_name = self.word2level[word_id][level_to_use]
                            cost = features.assign_search_cost(level_name)
                            new_candidate = (
                                point_3d_index,
                                level_name,
                                None,
                                cost,
                                "point"
                            )
                            features_to_match.append(new_candidate)
            elif candidate[-1] == "point":
                point_3d_index, level_name = candidate[:2]
                res_ = features.search(level_name, self.point_cloud[point_3d_index].desc)
                ans = self.point_cloud[point_3d_index]
                if res_ is not None:
                    feature_ind, dist, desc_2d = res_
                    pair = (feature_ind, point_3d_index, dist)
                    take_this = self.enforce_consistency(pair)

                    if debug and take_this:
                        ref_res = self.point_cloud.matching_2d_to_3d_brute_force(desc_2d)
                        samples += 1
                        if ref_res is not None and ans == ref_res:
                            count += 1

        result = []
        for f_id in self.matches:
            p_id, _ = self.matches[f_id]
            result.append((features[f_id], self.point_cloud[p_id]))
        print(f"Found {len(self.matches)} 2D-3D pairs, {len(features_to_match)} pairs left to consider.")
        return result, count, samples


if __name__ == '__main__':
    from feature_matching import build_descriptors_2d, load_2d_queries_opencv
    from colmap_io import read_points3D_coordinates, read_images, read_cameras
    from point2d import FeatureCloud
    from point3d import PointCloud

    query_images_folder = "test_images"
    cam_info_dir = "sfm_models/cameras.txt"
    sfm_images_dir = "sfm_models/images.txt"
    sfm_point_cloud_dir = "sfm_models/points3D.txt"
    sfm_images_folder = "sfm_models/images"
    camid2params = read_cameras(cam_info_dir)
    image2pose = read_images(sfm_images_dir)
    point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)
    points_3d_list = []
    point3d_id_list, point3d_desc_list = build_descriptors_2d(image2pose, sfm_images_folder)
    desc_list, coord_list, im_name_list = load_2d_queries_opencv(query_images_folder)

    point3d_cloud = PointCloud(debug=True)
    for i in range(len(point3d_id_list)):
        point3d_id = point3d_id_list[i]
        point3d_desc = point3d_desc_list[i]
        xyzrgb = point3did2xyzrgb[point3d_id]
        point3d_cloud.add_point(point3d_id, point3d_desc, xyzrgb[:3], xyzrgb[3:])
    point3d_cloud.commit()
    vocab_tree = VocabTree(point3d_cloud)
    desc_list, coord_list = desc_list[0], coord_list[0]
    feature_cloud = FeatureCloud()
    for i in range(len(desc_list)):
        feature_cloud.add_point(i, desc_list[i], coord_list[i])
    feature_cloud.assign_words(vocab_tree.word2level, vocab_tree.v1)
    res, c, s = vocab_tree.search(feature_cloud, debug=True)
    print(c, s)
    print(c/s)