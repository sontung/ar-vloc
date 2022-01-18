import numpy as np
import itertools
import pnp.build.pnp_python_binding
from tqdm import tqdm


def compute_smoothness_cost_pnp(solution, pid_coord_list, fid_coord_list):
    f, c1, c2 = 2600.0, 1134.0, 2016.0
    object_points = []
    image_points = []
    object_points_homo = []
    for u, v in enumerate(solution):
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
        cost += unary_cost_mat[u, v]
    return cost


def exhaustive_search(pid_desc_list, fid_desc_list, pid_coord_list, fid_coord_list):
    dim_x = pid_desc_list.shape[0]
    dim_y = fid_desc_list.shape[0]
    if dim_y > 7 or dim_x > 7:
        return
    unary_cost_mat = np.zeros((dim_x, dim_y), np.float64)
    for u in range(dim_x):
        for v in range(dim_y):
            unary_cost_mat[u, v] = np.sum(np.square(pid_desc_list[u] - fid_desc_list[v]))
    possible_solutions = list(itertools.permutations(range(dim_y), dim_x))
    min_cost = None
    best_solution = None
    for solution in tqdm(possible_solutions, desc="Exhaustive search"):
        c1 = compute_unary_cost(solution, unary_cost_mat)
        c2 = compute_smoothness_cost_pnp(solution, pid_coord_list, fid_coord_list)
        cost = c1 + c2
        if min_cost is None or cost < min_cost:
            min_cost = cost
            best_solution = solution
    print(min_cost, best_solution)
    return


if __name__ == '__main__':
    exhaustive_search([0, 1, 2, 3], [0, 1, 2, 3])
    # list1 = [0, 1, 2, 3, 4]
    # list2 = [0, 1, 2, 3, 4]
    # res = [list(zip(list2, x)) for x in itertools.permutations(list1, len(list2))]
    # for c in res:
    #     print([du[1] for du in c])
    # print(len(res))
    # print(len(list(itertools.permutations(list1, len(list2)))))
