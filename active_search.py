import numpy as np
import sklearn.cluster
import time
from scipy.spatial import KDTree


def build_vocabulary_of_descriptors(p3d_id_list, p3d_desc_list):
    nb_clusters = len(p3d_desc_list) // 5
    vocab = {u: [] for u in range(nb_clusters)}
    p3d_desc_list = np.array(p3d_desc_list)
    cluster_model = sklearn.cluster.KMeans(nb_clusters)
    labels = cluster_model.fit_predict(p3d_desc_list)
    for i in range(len(p3d_id_list)):
        vocab[labels[i]].append((p3d_id_list[i], p3d_desc_list[i]))
    return vocab, cluster_model


def matching_3d_to_2d_vocab_based(point3d_id_list, point3d_desc_list, point2d_desc_list, cluster_model, vocab):
    """
    returns [image id] => point 2d id => point 3d id
    """
    result = {i: [] for i in range(len(point2d_desc_list))}

    for i in range(len(point2d_desc_list)):
        desc_list = point2d_desc_list[i]

        # assign each desc to a word
        desc_list = np.array(desc_list)
        words = cluster_model.predict(desc_list)

        # sort feature by search cost
        features_to_match = [(du, desc_list[du], len(vocab[words[du]]), vocab[words[du]])
                             for du in range(desc_list.shape[0])]
        features_to_match = sorted(features_to_match, key=lambda du: du[2])

        for j, desc, _, point_3d_list in features_to_match:
            qu_point_3d_desc_list = [du2[1] for du2 in point_3d_list]
            qu_point_3d_id_list = [du2[0] for du2 in point_3d_list]

            kd_tree = KDTree(qu_point_3d_desc_list)
            res = kd_tree.query(desc, 2)
            if res[0][1] > 0.0:
                if res[0][0] / res[0][1] < 0.7:  # ratio test
                    result[i].append([j, qu_point_3d_id_list[res[1][0]]])

            if len(result[i]) >= 100:
                break
    return result


def matching_active_search(point3did2desc, point3did2xyzrgb,
                           point2d_desc_list, cluster_model, vocab):
    """
    returns [image id] => point 2d id => point 3d id
    """
    start_time = time.time()
    result = {i: [] for i in range(len(point2d_desc_list))}
    matching_acc = []

    # build tree for 3D points
    coordinates = []
    point3d_id_list = []
    for point3d in point3did2xyzrgb:
        xyz = point3did2xyzrgb[point3d][:3]
        coordinates.append(xyz)
        point3d_id_list.append(point3d)
    xyz_tree = KDTree(coordinates)

    for i in range(len(point2d_desc_list)):
        desc_list = point2d_desc_list[i]

        # build vocab for 2D descriptors
        vocab_active, cluster_model_active = build_vocabulary_of_descriptors(list(range(desc_list.shape[0])), desc_list)

        # assign each desc to a word
        desc_list = np.array(desc_list)
        words = cluster_model.predict(desc_list)

        # sort feature by search cost
        features_to_match = [(du, desc_list[du], len(vocab[words[du]]), vocab[words[du]])
                             for du in range(desc_list.shape[0])]
        features_to_match = sorted(features_to_match, key=lambda du: du[2])
        points3d_to_match = []

        count = 0
        for j, desc, _, point_3d_list in features_to_match:
            qu_point_3d_desc_list = [du2[1] for du2 in point_3d_list]
            qu_point_3d_id_list = [du2[0] for du2 in point_3d_list]

            kd_tree = KDTree(qu_point_3d_desc_list)
            res = kd_tree.query(desc, 2)
            if res[0][1] > 0.0:
                if res[0][0]/res[0][1] < 0.7:  # ratio test
                    result[i].append([j, qu_point_3d_id_list[res[1][0]]])
                    xyz = point3did2xyzrgb[qu_point_3d_id_list[res[1][0]]][:3]
                    _, neighbors = xyz_tree.query(xyz, 5)
                    for idx in neighbors:
                        point3d_id = point3d_id_list[idx]
                        point3d_desc = point3did2desc[point3d_id]
                        label = cluster_model_active.predict(point3d_desc)


            if len(result[i]) >= 100:
                matching_acc.append(count)
                break
    time_spent = time.time()-start_time
    print(f"Matching 2D-3D done in {round(time_spent, 3)} seconds, "
          f"avg. {round(time_spent/len(point2d_desc_list), 3)} seconds/image")
    return result
