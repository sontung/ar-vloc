import numpy as np
import itertools
import pnp.build.pnp_python_binding
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def prepare_neighbor_information(coord_list):
    nb_clusters = coord_list.shape[0] // 10
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


def prepare_input(pid_desc_list, fid_desc_list, pid_coord_list, fid_coord_list, correct_pairs=None):
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
            unary_cost_mat[u, v] = np.sum(np.square(pid_desc_list[u] - fid_desc_list[v]))
    pairwise_cost_mat = {}
    choices = list(range(len(edge_map)))
    del choices[0]
    for edge_id, (u0, v0) in enumerate(edge_map):
        for edge_id2 in choices:
            u1, v1 = edge_map[edge_id2]
            if u1 in n0[u0] and v1 in n1[v0]:
                pairwise_cost_mat[(edge_id, edge_id2)] = 0
        if len(choices) > 0:
            del choices[0]
    for (edge_id, edge_id2) in pairwise_cost_mat:
        assert (edge_id2, edge_id) not in pairwise_cost_mat
    print(len(pairwise_cost_mat))


def compute_smoothness_cost_pnp(solution, pid_coord_list, fid_coord_list):
    f, c1, c2 = 2600.0, 1134.0, 2016.0
    object_points = []
    image_points = []
    object_points_homo = []
    for u, v in enumerate(solution):
        if v >= 0:
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
    mat = pnp.build.pnp_python_binding.pnp(object_points, image_points)

    xy = mat[None, :, :] @ object_points_homo[:, :, None]
    xy = xy[:, :, 0]
    xy = xy[:, :3] / xy[:, 2].reshape((-1, 1))
    xy = xy[:, :2]
    diff = np.sum(np.square(xy - image_points), axis=1)
    inliers = np.sum(diff < 0.1)
    return inliers


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

    prepare_input(np.random.random((30, 128)),
                  np.random.random((30, 128)),
                  np.random.random((30, 3)),
                  np.random.random((30, 2)),
                  )
    # exhaustive_search([0, 1, 2, 3], [0, 1, 2, 3])
    # list1 = [0, 1, 2, 3, 4]
    # list2 = [0, 1, 2, 3, 4]
    # res = [list(zip(list2, x)) for x in itertools.permutations(list1, len(list2))]
    # for c in res:
    #     print([du[1] for du in c])
    # print(len(res))
    # print(len(list(itertools.permutations(list1, len(list2)))))