import numpy as np
import pnp.build.pnp_python_binding

import imageio
import cv2
import random
from colmap_io import read_points3D_coordinates
from feature_matching import compute_kp_descriptors_opencv_with_d2_detector, load_2d_queries_generic


def produce_data():
    with open('debug/test_refine.npy', 'rb') as afile:
        xyz_array = np.load(afile)
        xy_array = np.load(afile)

    new_coords = []
    f, c1, c2 = 2600.0, 1134.0, 2016.0
    xyz_array = np.reshape(xyz_array, (-1, 3))

    for ind in range(xy_array.shape[0]):
        x1, y1 = xy_array[ind, :]
        u = (x1 - c1) / f
        v = (y1 - c2) / f
        new_coords.append([u, v, 1])
    new_coords = np.array(new_coords)
    np.savetxt("/home/sontung/work/gosma/release/image.txt", new_coords)
    np.savetxt("/home/sontung/work/gosma/release/object.txt", xyz_array)


def process_results():
    res_line = "9.234939575e+00  7.089663148e-01 -2.337848663e+00  6.520420313e-01  2.970860600e-01 -6.975536942e-01 -1.082430407e-01  9.470853806e-01  3.021801114e-01  7.504163384e-01 -1.215288118e-01  6.496969461e-01 8.223325759e-02 1.534432200e-01 2.972210000e-04 1"
    numbers = []
    for du in res_line.split(" "):
        try:
            n = float(du)
        except ValueError:
            continue
        else:
            numbers.append(n)
    trans_mat = np.identity(4).astype(np.float64)
    trans_mat[0, 3] = -numbers[0]
    trans_mat[1, 3] = -numbers[1]
    trans_mat[2, 3] = -numbers[2]
    rot_mat = np.identity(4).astype(np.float64)
    rot_mat[:3, 0] = numbers[3:6]
    rot_mat[:3, 1] = numbers[6:9]
    rot_mat[:3, 2] = numbers[9:12]
    res_mat = rot_mat@trans_mat

    image_mat = np.loadtxt("/home/sontung/work/gosma/release/image.txt")
    object_mat = np.loadtxt("/home/sontung/work/gosma/release/object.txt")

    results = []
    for idx, xyz in enumerate(object_mat):
        xyz2 = rot_mat[:3, :3]@(xyz-numbers[:3])

        xyz = np.hstack([xyz, 1])
        xyz = res_mat@xyz

        u, v, w = xyz[:3]
        u /= w
        v /= w
        uv = np.array([u, v, 1])
        test = image_mat-uv
        test = test[:, :2]
        test = np.square(test)
        test = np.sum(test, axis=1)
        results.append([idx, np.min(test), np.argmin(test)])

    object_points = []
    image_points = []
    for ind in range(0, len(results), 5):
        sub_res = results[ind: ind+5]
        assert len(sub_res) == 5
        pid, _, fid = min(sub_res, key=lambda du: du[1])
        object_points.append(object_mat[pid])
        image_points.append(image_mat[fid, :2])
    object_points = np.array(object_points)
    image_points = np.array(image_points)
    res = pnp.build.pnp_python_binding.pnp(object_points, image_points)

    # return in opencv format
    r_mat, t_vec = res[:3, :3], res[:3, 3]
    t_vec = t_vec.reshape((-1, 1))
    return r_mat, t_vec


if __name__ == '__main__':
    produce_data()
    # process_results()