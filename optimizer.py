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


def prepare_neighbor_information(coord_list):
    nb_clusters = coord_list.shape[0] // 10
    if coord_list.shape[0] < 10:
        nb_clusters = 1
    cluster_model = MiniBatchKMeans(nb_clusters, random_state=1)
    labels = cluster_model.fit_predict(coord_list)
    label2index = {label: [] for label in range(nb_clusters)}
    for index, label in enumerate(labels):
        label2index[label].append(index)

    res = {}
    for label in label2index:
        neighborhood = label2index[label]
        for index in neighborhood:
            res[index] = []
            for index2 in neighborhood:
                if index2 != index:
                    res[index].append(index2)
    for index in range(coord_list.shape[0]):
        assert index in res
    return res


def prepare_input(pid_desc_list, fid_desc_list, pid_coord_list, fid_coord_list,
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
    n0 = prepare_neighbor_information(pid_coord_list)
    n1 = prepare_neighbor_information(fid_coord_list)
    dim_x = pid_desc_list.shape[0]
    dim_y = fid_desc_list.shape[0]
    unary_cost_mat = np.zeros((dim_x, dim_y), np.float64)
    edge_map = []
    for u in range(dim_x):
        for v in range(dim_y):
            edge_map.append((u, v))
            unary_cost_mat[u, v] = -np.sum(np.square(pid_desc_list[u] - fid_desc_list[v]))

    # normalize unary cost
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
            if only_neighbor:
                cond = u1 in n0[u0] and v1 in n1[v0]
            else:
                cond = True
            if cond:
                if zero_pairwise_cost:
                    pairwise_cost_mat[(edge_id, edge_id2)] = 1.0
                else:
                    cost = compute_pairwise_edge_cost(pid_coord_list[u0], fid_coord_list[v0],
                                                       pid_coord_list[u1], fid_coord_list[v1])
                    if cost == 0.0:
                        print(f" [warning]: very small edge cost: cost={cost} indices={v0, v1}"
                              f" coordinates={(fid_coord_list[v0][0], fid_coord_list[v0][1]), (fid_coord_list[v1][0], fid_coord_list[v1][1])}")
                    else:
                        pairwise_cost_mat[(edge_id, edge_id2)] = cost
        if len(choices) > 0:
            del choices[0]
    all_costs = list(pairwise_cost_mat.values())
    max_val, min_val = np.max(all_costs), np.min(all_costs)
    print(f" pairwise cost: max={max_val}, min={min_val}")
    for (edge_id, edge_id2) in pairwise_cost_mat:
        assert (edge_id2, edge_id) not in pairwise_cost_mat

    all_costs = list(pairwise_cost_mat.values())
    max_val, min_val = np.max(all_costs), np.min(all_costs)
    print(f" pairwise cost after normalized: max={max_val}, min={min_val}")
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


def compute_pairwise_edge_cost(u1, v1, u2, v2):
    v1 = np.array([v1[0], v1[1], 1.0])
    v2 = np.array([v2[0], v2[1], 1.0])
    vec1 = v1-u1
    vec2 = v2-u2
    return 1-cosine(vec1, vec2)


def compute_pairwise_edge_cost2(u1, v1, u2, v2):
    return np.sqrt(np.sum(np.square(v1-v2)))


def run_qap(pid_list, fid_list,
            pid_desc_list, fid_desc_list,
            pid_coord_list, fid_coord_list,
            correct_pairs, debug=False, qap_skip=True):
    if not qap_skip:
        prepare_input(pid_desc_list, fid_desc_list, pid_coord_list, fid_coord_list, correct_pairs)
        print(" running qap command")
        process = subprocess.Popen(["./run_qap.sh"], shell=True, stdout=subprocess.PIPE)
        process.wait()
        print(" done")
    else:
        print(" skipping qap optimization")
    labels = read_output(pid_coord_list, fid_coord_list)

    cost, object_points, image_points = compute_smoothness_cost_pnp(labels, pid_coord_list, fid_coord_list)
    if cost < 0.8*len(labels):
        return []
    solutions = []
    for idx in range(len(object_points)):
        u, v = object_points[idx], image_points[idx]
        solutions.append((pid_list[u], fid_list[v]))
    print(f" inlier cost={cost}, return {len(solutions)} matches")

    if debug:
        prepare_input(pid_desc_list, fid_desc_list, pid_coord_list, fid_coord_list, correct_pairs,
                      zero_pairwise_cost=True)
        process = subprocess.Popen(["./run_qap.sh"], shell=True, stdout=subprocess.PIPE)
        process.wait()
        labels_debug = read_output(pid_coord_list, fid_coord_list)
        cost, object_points2, image_points2 = compute_smoothness_cost_pnp(labels_debug, pid_coord_list, fid_coord_list)

        # agreement
        assert len(labels_debug) == len(labels)
        agree = 0
        distances = []
        for u, v in enumerate(labels_debug):
            if v == labels[u]:
                agree += 1
            if labels[u] is not None:
                coord1 = fid_coord_list[labels[u]]
                coord2 = fid_coord_list[v]
                distances.append(np.sum(np.abs(coord1-coord2))/coord2.shape[0])

        print(f" debug mode:")
        print(f"\t inlier cost={cost}")
        print(f"\t agreement={agree/len(labels)}, mean distance={np.mean(distances)}")

    return solutions


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
            u = (x - c1) / f
            v = (y - c2) / f
            image_points.append([u, v])
            object_points.append(xyz)
            x, y, z = xyz
            object_points_homo.append([x, y, z, 1.0])
    object_points = np.array(object_points)
    object_points_homo = np.array(object_points_homo)
    image_points = np.array(image_points)
    object_indices = np.array(object_indices)
    image_indices = np.array(image_indices)

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


if __name__ == '__main__':
    read_output()
    # prepare_input(np.random.random((30, 128)),
    #               np.random.random((30, 128)),
    #               np.random.random((30, 3)),
    #               np.random.random((30, 2)),
    #               )
    # exhaustive_search([0, 1, 2, 3], [0, 1, 2, 3])
    # list1 = [0, 1, 2, 3, 4]
    # list2 = [0, 1, 2, 3, 4]
    # res = [list(zip(list2, x)) for x in itertools.permutations(list1, len(list2))]
    # for c in res:
    #     print([du[1] for du in c])
    # print(len(res))
    # print(len(list(itertools.permutations(list1, len(list2)))))
