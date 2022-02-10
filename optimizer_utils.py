import itertools
import sys

from sklearn.cluster import MiniBatchKMeans
import numpy as np
import scipy.spatial
import pnp_utils


def smallest_set_containing(to_be_contained, universe):
    choices = list(itertools.product([0, 1], repeat=len(universe)))

    choice_tracks = {}
    for choice in choices:
        if np.sum(choice) <= 1:
            continue
        matches = []
        for u, v in enumerate(choice):
            if v > 0:
                matches.extend(universe[u])
        score = len(list(set(matches) & set(to_be_contained)))
        score2 = len([du for du in to_be_contained if du in matches])
        print(choice, score, score2)
    return


def prepare_neighbor_information(coord_list, nb_neighbors=10):
    tree = scipy.spatial.KDTree(coord_list)
    res = {}

    for i in range(coord_list.shape[0]):
        distances, _ = tree.query(coord_list[i], nb_neighbors)
        neighborhood = tree.query_ball_point(coord_list[i], distances[-1])
        res[i] = []
        for index in neighborhood:
            if i != index:
                res[i].append(index)
    for index in range(coord_list.shape[0]):
        assert index in res
    return res


def most_common_numbers(number2occurrence):
    arr = []
    keys = []
    for number in number2occurrence:
        keys.append(number)
        arr.append(number2occurrence[number])

    arr = np.array(arr).reshape(-1, 1)
    cluster_model = MiniBatchKMeans(2, random_state=1)
    labels = cluster_model.fit_predict(arr)
    most_common = np.argmax(cluster_model.cluster_centers_.reshape(-1, 1))
    res = [number for idx, number in enumerate(keys) if labels[idx] == most_common]
    return res


def compare_two_labels(label1, label2, fid_coord_list):
    agree = 0
    distances = []
    for u, v in enumerate(label1):
        if v == label2[u]:
            agree += 1
        if label2[u] is not None and v is not None:
            coord1 = fid_coord_list[label2[u]]
            coord2 = fid_coord_list[v]
            distances.append(np.sum(np.abs(coord1 - coord2)) / coord2.shape[0])
    acc = agree / len(label1)
    mean_distance = np.mean(distances)
    return acc, mean_distance


def compute_smoothness_cost_geometric(solution, pid_coord_list, fid_coord_list):
    object_points = []
    image_points = []
    for u, v in enumerate(solution):
        if v is not None:
            xyz = pid_coord_list[u]
            xy = fid_coord_list[v]
            image_points.append(xy)
            object_points.append(xyz)
    object_points = np.array(object_points)
    image_points = np.array(image_points)
    min_var_axis = min([0, 1, 2], key=lambda du: np.var(object_points[:, du]))
    object_points_projected = object_points[:, [ax for ax in range(3) if ax != min_var_axis]]
    diff = object_points_projected-image_points
    return np.var(np.abs(diff))


def evaluate(point2d_cloud, labels, pid_list, fid_list,
             pid_coord_list, fid_coord_list, f, c1, c2, against_gt=True):
    geom_cost = compute_smoothness_cost_geometric(labels, pid_coord_list, fid_coord_list)
    cost, object_points, image_points = pnp_utils.compute_smoothness_cost_pnp(labels, pid_coord_list, fid_coord_list, f, c1, c2)
    solutions = []
    for idx in range(len(object_points)):
        u, v = object_points[idx], image_points[idx]
        solutions.append((pid_list[u], fid_list[v]))

    # compare against gt
    res_gt = [(0, 213), (1, 98), (2, 47), (3, 169), (4, 88), (5, 218), (6, 229), (7, 220), (8, 191), (9, 237), (10, 204),
              (11, 188), (12, 7), (13, 48), (14, 14), (15, 211), (16, 186), (17, 112), (18, 97), (19, 129), (20, 173),
              (21, 104), (22, 77), (23, 222), (24, 39), (25, 13), (26, 119), (27, 56), (28, 66), (29, 31), (30, 115),
              (31, 159), (32, 29), (33, 189), (34, 87), (35, 163), (36, 184), (37, 170)]

    solutions_gt = [(3337, 455), (5004, 5865), (4241, 8781), (3602, 9089), (4374, 5848), (4000, 9170), (4001, 8166),
                    (4004, 9173), (3242, 9124), (4907, 7674), (3243, 4024), (3244, 9121), (4142, 7693), (4141, 4174),
                    (5168, 7702), (4144, 453), (4018, 9119), (4019, 5896), (4017, 8935), (4021, 5940), (4022, 9090),
                    (4535, 9207), (3251, 4267), (3527, 9175), (3400, 4164), (4167, 4117), (3282, 8486), (9810, 4190),
                    (3544, 649), (4056, 8760), (4057, 5914), (3546, 7009), (4058, 8756), (3303, 928), (3306, 4307),
                    (4205, 874), (4206, 6044), (8178, 385)]
    optimal_labels = [0 for _ in res_gt]
    for u, v in res_gt:
        optimal_labels[u] = v
    agree = 0
    distances = []
    acc = -1
    mean_distance = -1
    if against_gt:
        solutions_pred = [(pid_list[u], fid_list[v]) for u, v in enumerate(labels) if v is not None]
        pid2fid_gt = {x: y for x, y in solutions_gt}
        for u, v in solutions_pred:
            if v == pid2fid_gt[u]:
                agree += 1
            if pid2fid_gt[u] is not None and v is not None:
                coord1 = point2d_cloud[pid2fid_gt[u]].xy
                coord2 = point2d_cloud[v].xy
                distances.append(np.sum(np.abs(coord1 - coord2)) / coord2.shape[0])
        acc = agree/len(labels)
        mean_distance = np.mean(distances)
    return geom_cost, cost, solutions, acc, mean_distance


if __name__ == '__main__':
    d1 = [(5504, 4242), (4740, 5969), (5166, 7752), (4912, 6008), (5331, 7043), (5048, 8195), (5592, 786), (6601, 7743), (4019, 547), (5168, 385), (3544, 504), (9810, 661), (4205, 928)]
    d2 = {0: [(3303, 6063), (4205, 9173), (5168, 9124), (8178, 385), (3527, 4381), (9810, 661), (3243, 3943), (3251, 6008), (3242, 7043), (4004, 579)], 1: [(4000, 8476), (4206, 1015), (4142, 530), (4144, 8150), (4241, 8166), (4018, 8497), (4019, 547), (4021, 9035), (4374, 9170), (4022, 907)], 2: [(5504, 4242), (4740, 5969), (6025, 1063), (5166, 7752), (4912, 6008), (5331, 7043), (4533, 9124), (5048, 8195), (5943, 7046), (5592, 786)], 3: [(9089, 657), (5063, 8494), (6601, 7743), (4523, 9173), (4908, 9089), (5555, 814), (4540, 5944), (4861, 579), (4510, 8516)]}
    d3 = {(5504, 6008): 590, (4740, 7743): 601, (6025, 1063): 119, (5166, 7059): 584, (4912, 4242): 591, (5331, 8795): 295, (4533, 9124): 120, (5048, 856): 609, (5592, 835): 609, (9089, 657): 157, (5063, 8494): 130, (6601, 7743): 383, (4523, 9173): 319, (4908, 9089): 340, (5555, 814): 314, (4540, 5944): 125, (4861, 579): 321, (4510, 8516): 101, (4241, 8166): 376, (4018, 8497): 411, (4206, 1015): 267, (3544, 7702): 229, (3400, 8494): 272, (4205, 9173): 163, (3527, 9124): 150, (4004, 579): 272, (3306, 6089): 201, (3251, 6063): 205, (4017, 657): 181, (3242, 928): 193, (4000, 8476): 207, (4142, 530): 294, (4144, 8150): 287, (4019, 547): 253, (4021, 9035): 223, (4374, 9170): 208, (4022, 907): 115, (3244, 6978): 131, (5943, 896): 328}

    smallest_set_containing(d1, d2)
