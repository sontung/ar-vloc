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
    print(nb_possibilities)
    key2res = {}
    for k in keys:
        indices = list(map(int, k.split(" ")))
        d11 = [d1[i, j, :] for i, j in enumerate(indices)]
        d21 = [d2[i, j, :] for i, j in enumerate(indices)]
        d11 = np.array(d11)
        d21 = np.array(d21)
        res = pnp.build.pnp_python_binding.pnp(d11, d21)
        key2res[k] = (res, d11, d21, 0)
    diff_list = []
    best_diff = None
    max_size = 6
    for i in range(6, d1.shape[0]):
        d12 = d1[i, :, :]
        d22 = d2[i, :, :]
        records = []
        for k in key2res:
            res, d11, d21, _ = key2res[k]
            for ind, (x, y, z) in enumerate(d12):
                xyz = np.array([x, y, z, 1])
                xy = res@xyz
                xy = xy[:3]
                xy /= xy[-1]
                xy = xy[:2]
                diff = np.sum(np.square(xy-d22[ind]))
                records.append((diff, ind, k))
                diff_list.append(diff)
                if diff < 0.001:
                    new_xyz = np.array([x, y, z])
                    d11 = np.vstack([d11, new_xyz])
                    d21 = np.vstack([d21, d22[ind]])
                    res = pnp.build.pnp_python_binding.pnp(d11, d21)
                    key2res[k] = (res, d11, d21, diff)
                    if d11.shape[0] > max_size:
                        max_size = d11.shape[0]
                    break

        diff, ind, k = min(records, key=lambda du: du[0])
        if best_diff is None or diff <= best_diff:
            best_diff = diff
            print(f"add to {diff}")
            _, d11, d21, _ = key2res[k]
            x, y, z = d12[ind]
            new_xyz = np.array([x, y, z])
            d11 = np.vstack([d11, new_xyz])
            d21 = np.vstack([d21, d22[ind]])
            res = pnp.build.pnp_python_binding.pnp(d11, d21)
            key2res[k] = (res, d11, d21, diff)
            if d11.shape[0] > max_size:
                max_size = d11.shape[0]

        print(i, max([key2res[c][1].shape[0] for c in key2res]), best_diff, max_size)
    for k in key2res:
        res, d11, d21, diff = key2res[k]
        if d11.shape[0] < max_size:
            continue
        print(res.tolist())
        print(diff)


if __name__ == '__main__':
    with open('debug/test_refine.npy', 'rb') as afile:
        xyz_array = np.load(afile)
        xy_array = np.load(afile)
    refine_matching_pairs(xyz_array, xy_array)
