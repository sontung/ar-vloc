from math import sqrt
import numpy as np


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
