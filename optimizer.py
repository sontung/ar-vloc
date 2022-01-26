import sys

import numpy as np
import itertools
import thinqpbo as tq
import subprocess
import pickle
import json
import pnp.build.pnp_python_binding
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cosine
from tqdm import tqdm
from scipy.spatial import KDTree
from utils import angle_between


def prepare_neighbor_information(coord_list, nb_neighbors=10):
    tree = KDTree(coord_list)
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


def prepare_input(min_var_axis, pid_desc_list, fid_desc_list, pid_coord_list, fid_coord_list,
                  correct_pairs=None, only_neighbor=True, zero_pairwise_cost=False):
    """
    c comment line
    p <N0> <N1> <A> <E>     // # points in the left image, # points in the right image, # assignments, # edges
    a <a> <i0> <i1> <cost>  // specify assignment
    e <a> <b> <cost>        // specify edge

    i0 <id> {xi} {yi}       // optional - specify coordinate of a point in the left image
    i1 <id> {xi} {yi}       // optional - specify coordinate of a point in the left image
    n0 <i> <j>              // optional - specify that points <i> and <j> in the left image are neighbors
    n1 <i> <j>              // optional - specify that points <i> and <j> in the right image are neighbors
    """
    n0 = prepare_neighbor_information(pid_coord_list, 10)
    n1 = prepare_neighbor_information(fid_coord_list, 20)
    dim_x = pid_desc_list.shape[0]
    dim_y = fid_desc_list.shape[0]
    unary_cost_mat = np.zeros((dim_x, dim_y), np.float64)
    edge_map = []
    for u in range(dim_x):
        for v in range(dim_y):
            edge_map.append((u, v))
            unary_cost_mat[u, v] = np.sum(np.square(pid_desc_list[u] - fid_desc_list[v]))

    # normalize unary cost
    max_val, min_val = np.max(unary_cost_mat), np.min(unary_cost_mat)
    print(f" unary cost: max={max_val}, min={min_val}")
    div = max_val-min_val
    unary_cost_mat = (unary_cost_mat-min_val)/div+1
    max_val, min_val = np.max(unary_cost_mat), np.min(unary_cost_mat)
    print(f" unary cost after normalized: max={max_val}, min={min_val}")
    unary_cost_mat = -unary_cost_mat
    if correct_pairs is not None:
        for u, v in correct_pairs:
            print(f" setting correct pair {u} {v}")
            unary_cost_mat[u, :] = 100
            unary_cost_mat[u, v] = -100

    # pairwise
    pairwise_cost_mat = {}
    choices = list(range(len(edge_map)))
    del choices[0]
    for edge_id, (u0, v0) in enumerate(tqdm(edge_map, desc=" computing pairwise costs")):
        for edge_id2 in choices:
            u1, v1 = edge_map[edge_id2]
            if u1 == u0 or v1 == v0:
                continue
            if only_neighbor:
                cond = u1 in n0[u0] and v1 in n1[v0]
            else:
                cond = True
            if cond:
                if zero_pairwise_cost:
                    pairwise_cost_mat[(edge_id, edge_id2)] = 1.0
                else:
                    cost = compute_pairwise_edge_cost(pid_coord_list[u0], fid_coord_list[v0],
                                                      pid_coord_list[u1], fid_coord_list[v1], min_var_axis)
                    if cost == 0.0:
                        print(f" [warning]: very small edge cost: cost={cost} indices={v0, v1}"
                              f" coordinates={(fid_coord_list[v0][0], fid_coord_list[v0][1]), (fid_coord_list[v1][0], fid_coord_list[v1][1])}")
                    else:
                        pairwise_cost_mat[(edge_id, edge_id2)] = cost
        if len(choices) > 0:
            del choices[0]
    all_costs = list(pairwise_cost_mat.values())
    max_val, min_val = np.max(all_costs), np.min(all_costs)
    div = max_val - min_val

    print(f" pairwise cost: max={max_val}, min={min_val}")
    for (edge_id, edge_id2) in pairwise_cost_mat:
        old_val = pairwise_cost_mat[(edge_id, edge_id2)]
        if not zero_pairwise_cost:
            pairwise_cost_mat[(edge_id, edge_id2)] = (old_val - min_val) / div + 1

    all_costs = list(pairwise_cost_mat.values())
    max_val, min_val = np.max(all_costs), np.min(all_costs)
    print(f" pairwise cost after normalized: max={max_val}, min={min_val}")
    print(f" problem size: {unary_cost_mat.shape[0]*unary_cost_mat.shape[1]} nodes, {len(pairwise_cost_mat)} edges")
    write_to_dd(unary_cost_mat, pairwise_cost_mat, n0, n1)


def read_output(pid_coord_list, fid_coord_list, name="/home/sontung/work/ar-vloc/qap/fused.txt"):
    a_file = open(name)
    data = json.load(a_file)
    a_file.close()
    solution = data["labeling"]
    res = []
    for u, v in enumerate(solution):
        if v is not None:
            xyz = pid_coord_list[u]
            xy = fid_coord_list[v]
            res.append((xy, xyz))
    with open("qap/extra.pkl", "wb") as a_file:
        pickle.dump(res, a_file)
    return data["labeling"]


def compute_pairwise_edge_cost(u1, v1, u2, v2, min_var_axis):
    u1 = np.array([u1[du] for du in [0, 1, 2] if du != min_var_axis])
    u2 = np.array([u2[du] for du in [0, 1, 2] if du != min_var_axis])

    vec1 = u1-v1
    vec2 = u2-v2
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cost2 = np.abs(norm2-norm1)/(norm2+norm1)

    return np.abs(angle_between(vec1, vec2))+cost2
    # return np.var([v1, v2])


def compute_pairwise_edge_cost2(u1, v1, u2, v2):
    return np.sqrt(np.sum(np.square(v1-v2)))


def run_qap(pid_list, fid_list,
            pid_desc_list, fid_desc_list,
            pid_coord_list, fid_coord_list,
            point2d_cloud, point3d_cloud,
            correct_pairs, debug=True, qap_skip=False, optimal_label=False):
    pid_coord_var = np.var(pid_coord_list, axis=0)
    min_var_axis = min([0, 1, 2], key=lambda du: pid_coord_var[du])
    if optimal_label:
        res = [(0, 213), (1, 98), (2, 47), (3, 169), (4, 88), (5, 218), (6, 229), (7, 220), (8, 191), (9, 237), (10, 204), (11, 188), (12, 7), (13, 48), (14, 14), (15, 211), (16, 186), (17, 112), (18, 97), (19, 129), (20, 173), (21, 104), (22, 77), (23, 222), (24, 39), (25, 13), (26, 119), (27, 56), (28, 66), (29, 31), (30, 115), (31, 159), (32, 29), (33, 189), (34, 87), (35, 163), (36, 184), (37, 170)]

        solutions = [(3337, 455), (5004, 5865), (4241, 8781), (3602, 9089), (4374, 5848), (4000, 9170), (4001, 8166), (4004, 9173), (3242, 9124), (4907, 7674), (3243, 4024), (3244, 9121), (4142, 7693), (4141, 4174), (5168, 7702), (4144, 453), (4018, 9119), (4019, 5896), (4017, 8935), (4021, 5940), (4022, 9090), (4535, 9207), (3251, 4267), (3527, 9175), (3400, 4164), (4167, 4117), (3282, 8486), (9810, 4190), (3544, 649), (4056, 8760), (4057, 5914), (3546, 7009), (4058, 8756), (3303, 928), (3306, 4307), (4205, 874), (4206, 6044), (8178, 385)]
        labels = [0 for _ in res]
        for u, v in res:
            labels[u] = v
        for count, (pid, fid) in enumerate(solutions):
            assert pid_list.index(pid) == res[count][0] and fid_list.index(fid) == res[count][1]
            assert labels[count] == fid_list.index(fid)

        print(" using ground truth labels")
    else:
        if not qap_skip:
            prepare_input(min_var_axis, pid_desc_list, fid_desc_list, pid_coord_list, fid_coord_list, correct_pairs,
                          zero_pairwise_cost=False)
            print(" running qap command")
            process = subprocess.Popen(["./run_qap.sh"], shell=True, stdout=subprocess.PIPE)
            process.wait()
            print(" done")
        else:
            print(" skipping qap optimization")
        labels = read_output(pid_coord_list, fid_coord_list)
    geom_cost, cost, solutions, acc, dis = evaluate(point2d_cloud, point3d_cloud, labels,
                                                    pid_list, fid_list, pid_coord_list, fid_coord_list)
    print(f" inlier cost={cost}, return {len(solutions)} matches")
    print(f" geom cost={geom_cost}")
    print(f" compared against GT: acc={acc} dis={dis}")

    if debug:
        prepare_input(min_var_axis, pid_desc_list, fid_desc_list, pid_coord_list, fid_coord_list, correct_pairs,
                      zero_pairwise_cost=True)
        process = subprocess.Popen(["./run_qap.sh"], shell=True, stdout=subprocess.PIPE)
        process.wait()
        labels_debug = read_output(pid_coord_list, fid_coord_list)
        geom_cost, cost, _, acc, dis = evaluate(point2d_cloud, point3d_cloud, labels_debug,
                                                pid_list, fid_list, pid_coord_list, fid_coord_list)

        print(f" debug mode:")
        print(f"\t inlier cost={cost}")
        print(f"\t geom cost={geom_cost}")
        print(f"\t compared against GT: acc={acc} dis={dis}")
        acc, dis = compare_two_labels(labels_debug, labels, fid_coord_list)
        print(f"\t compared when w/ pw: acc={acc} dis={dis}")

    return solutions


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


def evaluate(point2d_cloud, point3d_cloud, labels, pid_list, fid_list,
             pid_coord_list, fid_coord_list, against_gt=True):
    geom_cost = compute_smoothness_cost_geometric(labels, pid_coord_list, fid_coord_list)
    cost, object_points, image_points = compute_smoothness_cost_pnp(labels, pid_coord_list, fid_coord_list)
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


def write_to_dd(unary_cost_mat, pairwise_cost_mat, pid_neighbors, fid_neighbors, name="qap/input.dd"):
    """
    c comment line
    p <N0> <N1> <A> <E>     // # points in the left image, # points in the right image, # assignments, # edges
    a <a> <i0> <i1> <cost>  // specify assignment
    e <a> <b> <cost>        // specify edge

    i0 <id> {xi} {yi}       // optional - specify coordinate of a point in the left image
    i1 <id> {xi} {yi}       // optional - specify coordinate of a point in the left image
    n0 <i> <j>              // optional - specify that points <i> and <j> in the left image are neighbors
    n1 <i> <j>              // optional - specify that points <i> and <j> in the right image are neighbors
    """
    a_file = open(name, "w")
    dim_x, dim_y = unary_cost_mat.shape
    print(f"p {dim_x} {dim_y} {dim_x*dim_y} {len(pairwise_cost_mat)}", file=a_file)
    assignment_id = 0
    for u in range(dim_x):
        for v in range(dim_y):
            print(f"a {assignment_id} {u} {v} {unary_cost_mat[u, v]}", file=a_file)
            assignment_id += 1
    for (edge_id, edge_id2) in pairwise_cost_mat:
        print(f"e {edge_id} {edge_id2} {pairwise_cost_mat[(edge_id, edge_id2)]}", file=a_file)
    for u in pid_neighbors:
        neighbors = pid_neighbors[u]
        for v in neighbors:
            print(f"n0 {u} {v}", file=a_file)
    for u in fid_neighbors:
        neighbors = fid_neighbors[u]
        for v in neighbors:
            print(f"n1 {u} {v}", file=a_file)
    a_file.close()


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


def compute_smoothness_cost_pnp(solution, pid_coord_list, fid_coord_list):
    f, c1, c2 = 2600.0, 1134.0, 2016.0
    object_points = []
    image_points = []
    object_points_homo = []
    object_indices = []
    image_indices = []

    for u, v in enumerate(solution):
        if v is not None:
            object_indices.append(u)
            image_indices.append(v)
            xyz = pid_coord_list[u]
            xy = fid_coord_list[v]
            x, y = xy
            u2 = (x - c1) / f
            v2 = (y - c2) / f
            image_points.append([u2, v2])
            object_points.append(xyz)
            x, y, z = xyz
            object_points_homo.append([x, y, z, 1.0])
    object_points = np.array(object_points)
    object_points_homo = np.array(object_points_homo)
    image_points = np.array(image_points)
    object_indices = np.array(object_indices)
    image_indices = np.array(image_indices)

    if object_points.shape[0] < 4:
        return -1, [], []
    mat = pnp.build.pnp_python_binding.pnp(object_points, image_points)

    xy = mat[None, :, :] @ object_points_homo[:, :, None]
    xy = xy[:, :, 0]
    xy = xy[:, :3] / xy[:, 2].reshape((-1, 1))
    xy = xy[:, :2]
    diff = np.sum(np.square(xy - image_points), axis=1)
    inliers = np.sum(diff < 0.1)
    object_indices, image_indices = object_indices[diff < 0.1], image_indices[diff < 0.1]
    return inliers, object_indices, image_indices


def compute_unary_cost(solution, unary_cost_mat):
    cost = 0
    for u, v in enumerate(solution):
        if v >= 0:
            cost += unary_cost_mat[u, v]
    return cost


def exhaustive_search(pid_desc_list, fid_desc_list, pid_coord_list, fid_coord_list, correct_pairs):
    dim_x = pid_desc_list.shape[0]
    dim_y = fid_desc_list.shape[0]
    if dim_y > 9 or dim_x > 9:
        return
    unary_cost_mat = np.zeros((dim_x, dim_y), np.float64)
    for u in range(dim_x):
        for v in range(dim_y):
            unary_cost_mat[u, v] = np.sum(np.square(pid_desc_list[u] - fid_desc_list[v]))
    for u, v in correct_pairs:
        unary_cost_mat[u, :] = 1000
        unary_cost_mat[u, v] = 0
    choices = list(range(dim_y))
    while len(choices) < dim_x:
        choices.append(-1)
    possible_solutions = list(itertools.permutations(choices, dim_x))
    min_cost = None
    best_solution = None
    for solution in possible_solutions:
        c1 = compute_unary_cost(solution, unary_cost_mat)
        c2 = compute_smoothness_cost_pnp(solution, pid_coord_list, fid_coord_list)
        cost = c1 + c2
        if min_cost is None or cost < min_cost:
            min_cost = cost
            best_solution = (solution, c1, c2)
    return best_solution


