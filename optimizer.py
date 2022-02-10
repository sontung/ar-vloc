import sys
import matplotlib.pyplot as plt
import numpy as np
import itertools
import subprocess
import pickle
import json
from optimizer_utils import most_common_numbers, prepare_neighbor_information, evaluate, compare_two_labels, \
    smallest_set_containing
from tqdm import tqdm
from utils import angle_between
from pnp_utils import compute_smoothness_cost_pnp, compute_smoothness_cost_pnp2
from itertools import product
from collections import Counter


def prepare_input(min_var_axis, pid_desc_list, fid_desc_list, pid_coord_list, fid_coord_list,
                  correct_pairs=None, only_neighbor=False, zero_pairwise_cost=False):
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
    unary_cost_mat_returned = np.copy(unary_cost_mat)
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
    return unary_cost_mat_returned


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


# @profile
def compute_pairwise_edge_cost(u1, v1, u2, v2, min_var_axis):
    u1 = np.array([u1[du] for du in [0, 1, 2] if du != min_var_axis])
    u2 = np.array([u2[du] for du in [0, 1, 2] if du != min_var_axis])

    vec1 = u1-v1
    vec2 = u2-v2

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    vec1 /= norm1
    vec2 /= norm2

    cost1 = angle_between(vec1, vec2)
    cost2 = np.abs(norm2-norm1)/(norm2+norm1)
    return cost1+cost2


# @profile
def run_qap(pid_list, fid_list,
            pid_desc_list, fid_desc_list,
            pid_coord_list, fid_coord_list,
            point2d_cloud,
            correct_pairs, f, c1, c2, debug=False, qap_skip=False, optimal_label=False):
    pid_coord_var = np.var(pid_coord_list, axis=0)
    min_var_axis = min([0, 1, 2], key=lambda du: pid_coord_var[du])
    pid_coord_list = normalize(pid_coord_list, 0)
    fid_coord_list = normalize(fid_coord_list, 0)

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
        total_pw_cost = compute_pairwise_cost_solution([(u, v) for u, v in enumerate(labels)],
                                                       pid_coord_list, fid_coord_list)
        solutions = [(3337, 455), (5004, 5865), (4241, 8781), (3602, 9089), (4374, 5848), (4000, 9170), (4001, 8166),
                     (4004, 9173), (3242, 9124), (4907, 7674), (3243, 4024), (3244, 9121), (4142, 7693), (4141, 4174),
                     (5168, 7702), (4144, 453), (4018, 9119), (4019, 5896), (4017, 8935), (4021, 5940), (4022, 9090),
                     (4535, 9207), (3251, 4267), (3527, 9175), (3400, 4164), (4167, 4117), (3282, 8486), (9810, 4190),
                     (3544, 649), (4056, 8760), (4057, 5914), (3546, 7009), (4058, 8756), (3303, 928), (3306, 4307),
                     (4205, 874), (4206, 6044), (8178, 385)]
        solutions_dict = {u: v for u, v in solutions}
        optimal_labels = [(pid_list.index(u), fid_list.index(solutions_dict[u])) for u in pid_list
                          if solutions_dict[u] in fid_list]
        total_pw_cost_optimal = compute_pairwise_cost_solution(optimal_labels, pid_coord_list, fid_coord_list)
        print(f" total pw cost={total_pw_cost}")
        print(f" total pw cost optimal={total_pw_cost_optimal}")

    geom_cost, cost, solutions, acc, dis = evaluate(point2d_cloud, labels,
                                                    pid_list, fid_list, pid_coord_list, fid_coord_list, f, c1, c2)
    print(f" inlier cost={cost}, return {len(solutions)} matches")
    print(f" geom cost={geom_cost}")
    print(f" compared against GT: acc={acc} dis={dis}")

    if debug:
        prepare_input(min_var_axis, pid_desc_list, fid_desc_list, pid_coord_list, fid_coord_list, correct_pairs,
                      zero_pairwise_cost=True)
        process = subprocess.Popen(["./run_qap.sh"], shell=True, stdout=subprocess.PIPE)
        process.wait()
        labels_debug = read_output(pid_coord_list, fid_coord_list)
        geom_cost, cost, _, acc, dis = evaluate(point2d_cloud, labels_debug,
                                                pid_list, fid_list, pid_coord_list, fid_coord_list, f, c1, c2)

        print(f" debug mode:")
        print(f"\t inlier cost={cost}")
        print(f"\t geom cost={geom_cost}")
        print(f"\t compared against GT: acc={acc} dis={dis}")
        acc, dis = compare_two_labels(labels_debug, labels, fid_coord_list)
        print(f"\t compared when w/ pw: acc={acc} dis={dis}")

    return solutions


def run_qap_final(pid_list, fid_list,
                  pid_desc_list, fid_desc_list,
                  pid_coord_list, fid_coord_list,
                  point2d_cloud,
                  correct_pairs, f, c1, c2, index):
    pid_coord_var = np.var(pid_coord_list, axis=0)
    min_var_axis = min([0, 1, 2], key=lambda du: pid_coord_var[du])
    pid_coord_list = normalize(pid_coord_list, 0)
    fid_coord_list = normalize(fid_coord_list, 0)

    unary_cost_mat = prepare_input(min_var_axis, pid_desc_list, fid_desc_list, pid_coord_list, fid_coord_list, correct_pairs,
                                   zero_pairwise_cost=False)
    print(" running qap command")
    process = subprocess.Popen(["./run_qap.sh"], shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(" done")
    labels = read_output(pid_coord_list, fid_coord_list)
    total_pw_cost = compute_pairwise_cost_solution([(u, v) for u, v in enumerate(labels)],
                                                   pid_coord_list, fid_coord_list)
    total_unary_cost = np.mean([unary_cost_mat[u, v] for u, v in enumerate(labels)])
    print(f" pw cost={total_pw_cost}, unary cost={total_unary_cost}")

    geom_cost, cost, solutions, _, _ = evaluate(point2d_cloud, labels,
                                                pid_list, fid_list,
                                                pid_coord_list, fid_coord_list, f, c1, c2, against_gt=False)
    for u, v in enumerate(labels):
        pid_coord = pid_coord_list[u]
        fid_coord = fid_coord_list[v]
        pid_coord_wo_min_axis = np.array([pid_coord[du] for du in [0, 1, 2] if du != min_var_axis])
        x1, y1 = [pid_coord_wo_min_axis[0]-2, fid_coord[0]+2], [pid_coord_wo_min_axis[1], fid_coord[1]]
        plt.plot(x1, y1, marker='o')
    plt.savefig(f"debug/pw/img-{index}.png")
    plt.close()

    if cost <= 0.7*len(pid_list):
        return []
    print(f" inlier cost={cost}, return {len(solutions)} matches")

    return solutions


def exhaustive_filter_post_optim(point3d_cloud, point2d_cloud, f, c1, c2, universe, debug=True):
    if len(universe) > 10:
        return []
    tracks = []
    choices = list(product([0, 1], repeat=len(universe)))
    best_cost = None
    choice_tracks = {}
    for choice in choices:
        if np.sum(choice) <= 1:
            continue
        choice_tracks[choice] = []

    choice2match = {}
    for choice in choice_tracks:
        matches = []
        for u, v in enumerate(choice):
            if v > 0:
                matches.extend(universe[u])
        votes = {}
        for _ in range(100):
            cost, p_indices, f_indices = compute_smoothness_cost_pnp2(matches, point3d_cloud, point2d_cloud, f, c1, c2)
            cost_norm = cost/len(matches)
            choice_tracks[choice].append([p_indices.tolist(), f_indices.tolist(), len(matches), cost, cost_norm])
            for idx, pid in enumerate(p_indices):
                fid = f_indices[idx]
                key_ = (pid, fid)
                if key_ in votes:
                    votes[key_] += 1
                else:
                    votes[key_] = 1
        print(choice)
        data_ = most_common_numbers(votes)
        print(f" {data_}")
        choice2match[choice] = matches

    return choice2match[(1, 1, 0, 0)]

    # # normalize all iterations
    # for choice in choice_tracks:
    #     if choice != (1, 1, 0, 0):
    #         continue
    #     res = choice_tracks[choice]
    #     all_costs = [du[-2] for du in res]
    #     most_common_costs = most_common_numbers(Counter(all_costs))
    #     cost = np.mean(most_common_costs)
    #     cost_norm = cost/res[0][-3]
    #     tracks.append([choice, res[0][0], res[0][1], most_common_costs,
    #                    [du[-1] for du in res], [du[-2] for du in res], cost, cost_norm])
    #     if best_cost is None or cost_norm > best_cost:
    #         best_cost = cost_norm
    #
    # tracks = sorted(tracks, key=lambda du: du[-1])
    # for track in tracks:
    #     print(track[0], track[-2], track[-1])
    #     print(f" {Counter(track[-3])}")
    #     print(f" {track[-5]}")
    #
    # # in case of tie, breaking by choosing the biggest number of inliers
    # best_tracks = [track for track in tracks if track[-1] == best_cost]
    # if len(best_tracks) == 1:
    #     best_track = best_tracks[0]
    # else:
    #     best_track = max(best_tracks, key=lambda du: du[-2])
    # p_indices = best_track[1]
    # f_indices = best_track[2]
    #
    # matches = []
    # assert len(p_indices) == len(f_indices)
    # for u, v in enumerate(p_indices):
    #     matches.append((v, f_indices[u]))
    # return matches


def write_to_dd(unary_cost_mat, pairwise_cost_mat, pid_neighbors, fid_neighbors, name="qap/input.dd",
                write_neighbor_info=True):
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
    if write_neighbor_info:
        for u in pid_neighbors:
            neighbors = pid_neighbors[u]
            for v in neighbors:
                print(f"n0 {u} {v}", file=a_file)
        for u in fid_neighbors:
            neighbors = fid_neighbors[u]
            for v in neighbors:
                print(f"n1 {u} {v}", file=a_file)
    a_file.close()


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


def normalize(coord_list, offset):
    x_distance = np.max(coord_list[:, 0]) - np.min(coord_list[:, 0])
    coord_list[:, 0] = (coord_list[:, 0] - np.min(coord_list[:, 0])) / x_distance
    for i in range(1, coord_list.shape[1]):
        distance = np.max(coord_list[:, i]) - np.min(coord_list[:, i])
        new_max = distance / x_distance
        coord_list[:, i] = (coord_list[:, i] - np.min(coord_list[:, i])) * new_max / distance

    return coord_list + offset


def compute_pairwise_cost_solution(label, cloud1, cloud2):
    pairwise_cost_mat = {}
    choices = list(range(len(label)))
    del choices[0]
    total_cost_ = 0
    count = 0
    for edge_id, (u0, v0) in enumerate(label):
        for edge_id2 in choices:
            u1, v1 = label[edge_id2]
            if u1 == u0 or v1 == v0:
                continue
            cost_ = compute_pairwise_edge_cost(cloud1[u0], cloud2[v0],
                                               cloud1[u1], cloud2[v1], 2)
            pairwise_cost_mat[(edge_id, edge_id2)] = cost_
            total_cost_ += cost_
            count += 1
        if len(choices) > 0:
            del choices[0]
    return total_cost_/count

