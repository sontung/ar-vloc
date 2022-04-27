import numpy as np
from numba import njit, prange


@njit("f8[:, :](f8[:, :, :], f8[:, :, :])")
def fast_sum(x, ty):
    res = np.sum((x - ty) ** 2, axis=2)
    return res


@njit("i1(i1[:])")
def fast_sum_i1(xx):
    res = np.sum(xx)
    return res


@njit("f4[:](f4[:], f4[:, :])")
def fast_sq_difference_f16(vec1, vec2):
    res = np.sum((vec1 - vec2)**2, axis=1)
    return res


@njit("f8[:, :](f8[:, :], f8[:, :])", parallel=True)
def fast_sum_parallel(x, ty):
    res = np.zeros((ty.shape[0], x.shape[0]))
    for i in prange(ty.shape[0]):
        res[i] = np.sum((x-ty[i])**2, axis=-1)
    return res


@njit("f8[:, :](f8[:, :], f8)", parallel=True)
def fast_exp(p, s):
    for i in prange(p.shape[0]):
        for j in prange(p.shape[1]):
            p[i, j] = np.exp(-p[i, j] / (2 * s))
    return p


def initialize_sigma2(X, Y):
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = X[None, :, :] - Y[:, None, :]
    err = diff ** 2
    return np.sum(err) / (D * M * N)


@njit("f8[:](f8[:, :])")
def fast_sum_axis0(x):
    return np.sum(x, axis=0)