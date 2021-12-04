import numpy as np
from sklearn.cluster import MiniBatchKMeans


class VocabNode:
    def __init__(self, nb_clusters):
        self.nb_clusters = nb_clusters
        self.vocab = None
        self.cluster_model = None
        return

    def build(self, p3d_id_list, p3d_desc_list):
        if self.nb_clusters > len(p3d_id_list):
            raise AttributeError
        self.vocab = {u: [] for u in range(self.nb_clusters)}
        p3d_desc_list = np.array(p3d_desc_list)
        self.cluster_model = MiniBatchKMeans(self.nb_clusters)
        labels = self.cluster_model.fit_predict(p3d_desc_list)
        for i in range(len(p3d_id_list)):
            self.vocab[labels[i]].append((p3d_id_list[i], p3d_desc_list[i], i))
        return


class VocabNodeForWords:
    def __init__(self, nb_clusters):
        self.nb_clusters = nb_clusters
        self.vocab = None
        self.cluster_model = None
        return

    def build(self, words):
        self.vocab = {u: [] for u in range(self.nb_clusters)}
        self.cluster_model = MiniBatchKMeans(self.nb_clusters)
        labels = self.cluster_model.fit_predict(words)
        for i in range(len(words)):
            self.vocab[labels[i]].append((words[i], i))
        return


class VocabTree:
    def __init__(self, p3d_id_list, p3d_desc_list, branching_factor=5):
        self.v1 = VocabNode(nb_clusters=500)
        self.v1.build(p3d_id_list, p3d_desc_list)
        self.v2 = VocabNode(nb_clusters=branching_factor)
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
    print(desc_list[0].shape)
    print(len(point3d_desc_list))
    vocab_tree = VocabTree(point3d_id_list, point3d_desc_list)
