import numpy as np
from sklearn.cluster import MiniBatchKMeans


class VocabNode:
    def __init__(self, nb_clusters):
        self.nb_clusters = nb_clusters
        self.vocab = None
        self.cluster_model = None

    def build(self, p3d_id_list, p3d_desc_list):
        if self.nb_clusters > len(p3d_id_list):
            raise AttributeError
        self.vocab = {u: [] for u in range(self.nb_clusters)}
        p3d_desc_list = np.array(p3d_desc_list)
        self.cluster_model = MiniBatchKMeans(self.nb_clusters)
        labels = self.cluster_model.fit_predict(p3d_desc_list)
        for i in range(len(p3d_id_list)):
            self.vocab[labels[i]].append((p3d_id_list[i], p3d_desc_list[i], i))
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
    def __init__(self, p3d_id_list, p3d_desc_list, branching_factor=2):
        self.v1 = VocabNode(nb_clusters=500)
        words = self.v1.build(p3d_id_list, p3d_desc_list)
        word_indices = [word.index for word in words]
        word_descriptors = [word.descriptor for word in words]

        self.v2 = VocabNodeForWords("0", 1, branching_factor)
        data = self.v2.build(word_indices, word_descriptors)

        self.word2level = {word_ind: {} for word_ind in word_indices}
        self.level2words = {}
        for word_ind, level, level_name in data:
            assert level not in self.word2level[word_ind]
            self.word2level[word_ind][level] = level_name
            if level_name not in self.level2words:
                self.level2words[level_name] = [word_ind]
            else:
                self.level2words[level_name].append(word_ind)

    def search(self, query_desc):
        return 


if __name__ == '__main__':
    from feature_matching import build_descriptors_2d, load_2d_queries_opencv
    from colmap_io import read_points3D_coordinates, read_images, read_cameras

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
    vocab_tree = VocabTree(point3d_id_list, point3d_desc_list)
