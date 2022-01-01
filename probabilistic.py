import sys
import random
import numpy as np
import pnp.build.pnp_python_binding
from tqdm import tqdm


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
    removed_each_iter = (len(keys)-1) // len(list(range(6, d1.shape[0])))
    removed_each_iter = 0
    key2res = {}
    for k in tqdm(keys, desc="Computing initial PNP seeds"):
        indices = list(map(int, k.split(" ")))
        d11 = [d1[i, j, :] for i, j in enumerate(indices)]
        d21 = [d2[i, j, :] for i, j in enumerate(indices)]
        d11 = np.array(d11)
        d21 = np.array(d21)
        res = pnp.build.pnp_python_binding.pnp(d11, d21)
        key2res[k] = (res, d11, d21, [])

    for i in range(6, d1.shape[0]):
        d12 = d1[i, :, :]
        d22 = d2[i, :, :]
        records = []
        mean_diff_list = []
        for k in key2res:
            res, d11, d21, matches = key2res[k]
            diff_list = []

            d13 = np.hstack([d12, np.ones((d12.shape[0], 1))])
            xy = res[None, :, :] @ d13[:, :, None]
            xy = xy[:, :, 0]
            xy = xy[:, :3] / xy[:, 2].reshape((-1, 1))
            xy = xy[:, :2]
            diff = np.sum(np.abs(xy - d22), axis=1)
            diff_list.extend([(k, i, ind, diff[ind]) for ind in range(diff.shape[0])])

            # for ind, (x, y, z) in enumerate(d12):
            #     xyz = np.array([x, y, z, 1])
            #     xy = res@xyz
            #     xy = xy[:3]
            #     xy /= xy[-1]
            #     xy = xy[:2]
            #     diff = np.sum(np.abs(xy-d22[ind]))
            #     diff_list.append((k, i, ind, diff))

            records.append(min(diff_list, key=lambda du: du[-1]))
            mean_diff_list.append(records[-1][-1])

        print(f"Iter {i}, candidates={len(key2res)}, mean diff={round(np.mean(mean_diff_list), 3)}")

        # removing
        records = sorted(records, key=lambda du1: du1[-1], reverse=True)
        assert records[0][-1] >= records[-1][-1]
        removed_list = records[:removed_each_iter]
        for k, _, ind, diff in removed_list:
            del key2res[k]

        # updating
        for k, i2, ind, diff in records[removed_each_iter:]:
            res, d11, d21, matches = key2res[k]
            matches.append((i2, ind))
            d11 = np.vstack([d11, d12[ind]])
            d21 = np.vstack([d21, d22[ind]])
            assert k in key2res
            key2res[k] = (res, d11, d21, matches)

    best_res = None
    best_diff = None
    for k in tqdm(key2res, desc="Final selection"):
        res, d11, d21, matches = key2res[k]
        res = pnp.build.pnp_python_binding.pnp(d11, d21)
        d11 = np.hstack([d11, np.ones((d11.shape[0], 1))])

        xy = res[None, :, :] @ d11[:, :, None]
        xy = xy[:, :, 0]
        xy = xy[:, :3] / xy[:, 2].reshape((-1, 1))
        xy = xy[:, :2]
        final_diff = np.sum(np.abs(d21-xy))/2/d21.shape[0]
        if best_res is None or final_diff < best_diff:
            best_diff = final_diff
            best_res = res
    print(best_res.tolist())
    print(best_diff)


def evaluate(s, d1, d2):
    d11 = d1[list(range(len(s))), s, :]
    d21 = d2[list(range(len(s))), s, :]
    mat = pnp.build.pnp_python_binding.pnp(d11, d21)
    d13 = np.hstack([d11, np.ones((d11.shape[0], 1))])
    xy = mat[None, :, :] @ d13[:, :, None]
    xy = xy[:, :, 0]
    xy = xy[:, :3] / xy[:, 2].reshape((-1, 1))
    xy = xy[:, :2]
    diff = np.sum(np.abs(xy - d21))/d11.shape[0]/2
    return diff


def transition_sa(state, d1, d2, state_cost, indices, actions, idx_prob, action_prob):
    new_state = state[:]
    idx = np.random.choice(indices, p=idx_prob)
    action = np.random.choice(actions, p=action_prob[idx])
    new_state[idx] = action
    new_cost = evaluate(new_state, d1, d2)

    # update probabilities
    if new_cost < state_cost:
        idx_prob[idx] /= 20
        action_prob[idx][action] *= 20
    # else:
    #     idx_prob[idx] *= 10
    #     action_prob[idx][action] /= 10

    # normalize probabilities
    norm = np.sum(action_prob[idx])
    if norm == 0:
        idx_prob[idx] = 0
    else:
        action_prob[idx] /= norm

    norm2 = np.sum(idx_prob)
    if norm2 > 0:
        idx_prob /= np.sum(idx_prob)

    return new_state, new_cost


def simulated_annealing(d1, d2, nb_iters=10000, steps_per_temp=10, termination=100, temp=10):
    print(d1.shape, d2.shape)
    indices = list(range(d1.shape[0]))
    actions = list(range(d1.shape[1]))
    idx_prob = np.ones((d1.shape[0],))*1/d1.shape[0]
    action_prob = np.ones((d1.shape[0], d1.shape[1]))*1/d1.shape[1]

    # exploring
    best_state = None
    best_cost = None
    state = [random.choice(actions) for _ in range(d1.shape[0])]
    current_cost = evaluate(state, d1, d2)

    for _ in range(100):
        new_state, new_cost = transition_sa(state, d1, d2, current_cost, indices, actions, idx_prob, action_prob)
        if best_cost is None or new_cost < best_cost:
            best_state = new_state
            best_cost = new_cost
    state = best_state
    current_cost = best_cost

    for iter_ in range(nb_iters):
        if np.sum(idx_prob) == 0:
            break
        new_state, new_cost = transition_sa(state, d1, d2, current_cost, indices, actions, idx_prob, action_prob)
        print(f"Iter {iter_}: best cost={current_cost} current cost={new_cost}")

        if new_cost < current_cost:
            current_cost = new_cost
            state = new_state


if __name__ == '__main__':
    with open('debug/test_refine.npy', 'rb') as afile:
        xyz_array = np.load(afile)
        xy_array = np.load(afile)
    simulated_annealing(xyz_array, xy_array)
