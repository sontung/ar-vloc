from math import sqrt
import numpy as np
import pydegensac
import copy
import itertools


def norm_squared(vector):
    return vector[0]*vector[0]+vector[1]*vector[1]+vector[2]*vector[2]


def norm(vector):
    return sqrt(norm_squared(vector))


def normalize_vector(vector):
    n = norm(vector)
    if abs(n) <= 0.0001:
        print("norm", n, vector)
    return vector/n


def multDirMatrix(src, x):
    a = src[0] * x[0][0] + src[1] * x[1][0] + src[2] * x[2][0]
    b = src[0] * x[0][1] + src[1] * x[1][1] + src[2] * x[2][1]
    c = src[0] * x[0][2] + src[1] * x[1][2] + src[2] * x[2][2]
    return np.array([a, b, c])


def geometric_verify_pydegensac(src_pts, dst_pts, th=4.0, n_iter=2000):
    h_mat, mask = pydegensac.findHomography(src_pts, dst_pts, th, 0.99, n_iter)
    nb_inliers = int(copy.deepcopy(mask).astype(np.float32).sum())
    return h_mat, mask, nb_inliers, nb_inliers / src_pts.shape[0]


def filter_pairs(ori_pairs, mask_):
    return [ori_pairs[idx] for idx in range(len(ori_pairs)) if mask_[idx]]


def ccw(p1, p2, p3):
    return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])


def intersect(p1, p2, p3, p4):
    """
    Return true if line segments p1 p2 and p3 p4 intersect
    """
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def quadrilateral_self_intersect_test(p1, p2, p3, p4):
    """
    check if a quadrilateral self-intersects
    """
    lines = [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]
    for x, y in itertools.combinations([0, 1, 2, 3], 2):
        u, v = lines[x]
        m, n = lines[y]
        if intersect(u, v, m, n):
            return True
    return False


if __name__ == '__main__':
    du = intersect([0, 0], [0, 2], [0, 2], [1, 2])
    print(du)
    # quadrilateral_self_intersect_test(0, 1, 2, 3)
    for x, y in itertools.combinations([0, 1, 2, 3], 2):
        print(x, y)
