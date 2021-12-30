import sys
import numpy as np
import pnp.build.pnp_python_binding


def refine_matching_pairs(d1, d2):
    """
    refine matching
    :param d1: xyz (world coordinates of 3D model)
    :param d2: xy (camera coordinates of keypoints)
    :return:
    """
    nb_possibilities = d1.shape[1]
    keys = []
    for a in range(nb_possibilities):
        for b in range(nb_possibilities):
            for c in range(nb_possibilities):
                for d in range(nb_possibilities):
                    for e in range(nb_possibilities):
                        for f in range(nb_possibilities):
                            keys.append(f"{a} {b} {c} {d} {e} {f}")

    key2res = {}
    for k in keys:
        indices = list(map(int, k.split(" ")))
        d11 = [d1[i, j, :] for i, j in enumerate(indices)]
        d21 = [d2[i, j, :] for i, j in enumerate(indices)]
        d11 = np.array(d11)
        d21 = np.array(d21)
        res = pnp.build.pnp_python_binding.pnp(d11, d21)
        key2res[k] = (res, d11, d21)

    for i in range(6, d1.shape[0]):
        d12 = d1[i, :, :]
        d22 = d2[i, :, :]
        deleted = []
        for k in key2res:
            res, d11, d21 = key2res[k]
            select = False
            for ind, (x, y, z) in enumerate(d12):
                xyz = np.array([x, y, z, 1])
                xy = res@xyz
                xy = xy[:3]
                xy /= xy[-1]
                xy = xy[:2]
                diff = np.sum(np.square(xy-d22[ind]))
                if diff < 0.001:
                    select = True
                    new_xyz = np.array([x, y, z])
                    d11 = np.vstack([d11, new_xyz])
                    d21 = np.vstack([d21, d22[ind]])
                    res = pnp.build.pnp_python_binding.pnp(d11, d21)
                    key2res[k] = (res, d11, d21)
                    break
            if not select:
                deleted.append(k)

        for k in deleted:
            del key2res[k]
        print(len(key2res))
    print(list(key2res.keys()))
    for k in key2res:
        res = key2res[k][0]
        print(res)


if __name__ == '__main__':
    with open('debug/test_refine.npy', 'rb') as f:
        xyz_array = np.load(f)
        xy_array = np.load(f)
    refine_matching_pairs(xyz_array, xy_array)
