from sklearn.cluster import MiniBatchKMeans
import numpy as np


def cluster_pos(point_cloud):
    p3d_id_list, p3d_desc_list = point_cloud.point_id_list, point_cloud.point_desc_list
    p3d_pos_list = point_cloud.point_xyz_list
    nb_clusters = len(point_cloud)//5
    if nb_clusters > len(p3d_id_list):
        raise AttributeError
    data = np.array(p3d_pos_list)
    cluster_model = MiniBatchKMeans(nb_clusters, random_state=1)
    labels = cluster_model.fit_predict(data)
    for ind in range(len(p3d_id_list)):
        point_cloud[ind].visual_word = labels[ind]