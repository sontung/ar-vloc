import numpy as np
import random
import sys
import pnp.build.pnp_python_binding
from tqdm import tqdm


def project(state_, d1, d2):
    d11 = d1[list(range(len(state_))), state_, :]
    d21 = d2[list(range(len(state_))), state_, :]
    mat = pnp.build.pnp_python_binding.pnp(d11, d21)
    d13 = np.hstack([d11, np.ones((d11.shape[0], 1))])
    xy = mat[None, :, :] @ d13[:, :, None]
    xy = xy[:, :, 0]
    xy = xy[:, :3] / xy[:, 2].reshape((-1, 1))
    xy = xy[:, :2]
    return xy


def project_raw(d11, d21):
    mat = pnp.build.pnp_python_binding.pnp(d11, d21)
    d13 = np.hstack([d11, np.ones((d11.shape[0], 1))])
    xy = mat[None, :, :] @ d13[:, :, None]
    xy = xy[:, :, 0]
    xy = xy[:, :3] / xy[:, 2].reshape((-1, 1))
    xy = xy[:, :2]
    diff = np.sum(np.square(xy - d21), axis=1)
    inliers = np.sum(diff < 0.1)
    print(np.nonzero(diff < 0.1)[0])
    err = np.sum(np.abs(xy-d21))/2/d11.shape[0]
    return inliers, err


def evaluate(state_, d1, d2):
    d21 = d2[list(range(len(state_))), state_, :]
    xy = project(state_, d1, d2)
    diff = np.sum(np.square(xy - d21), axis=1)
    inliers = np.sum(diff < 0.1)
    return inliers


def optimize(state_, d1, d2):
    max_action = d1.shape[1]
    action_list = list(range(max_action))
    action_list.remove(state_[0])
    inliers, ori_err = evaluate(state_, d1, d2)
    print(inliers, action_list, ori_err)
    ori = len(inliers)
    for idx in tqdm(range(len(state_)), desc="Optimizing"):
        if idx in inliers:
            continue
        new_state = state_[:]
        for opt in action_list:
            new_state[idx] = opt
            res, new_err = evaluate(new_state, d1, d2)
            if len(res) > ori:
                ori = len(res)
                state_[idx] = opt
                print(f"Idx={idx} choice={opt} new_err={new_err}")
    e1, e2 = evaluate(state_, d1, d2)
    print("final", e1, e2)


def transition_sa(state, d1, d2, state_cost, indices, actions, idx_prob, action_prob):
    new_state = state[:]
    idx = np.random.choice(indices, p=idx_prob)
    action = np.random.choice(actions, p=action_prob[idx])
    new_state[idx] = action
    new_cost = evaluate(new_state, d1, d2)

    # update probabilities
    if new_cost > state_cost:
        idx_prob[idx] /= 20
        action_prob[idx][action] *= 20

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
    indices = list(range(d1.shape[0]))
    actions = list(range(d1.shape[1]))
    idx_prob = np.ones((d1.shape[0],))*1/d1.shape[0]
    action_prob = np.ones((d1.shape[0], d1.shape[1]))*1/d1.shape[1]

    # exploring
    best_state = None
    best_cost = None

    for _ in range(100):
        state = [random.choice(actions) for _ in range(d1.shape[0])]
        current_cost = evaluate(state, d1, d2)
        if best_cost is None or current_cost > best_cost:
            best_state = state
            best_cost = current_cost
    state = best_state
    current_cost = best_cost

    for iter_ in range(nb_iters):
        if np.sum(idx_prob) == 0:
            break
        new_state, new_cost = transition_sa(state, d1, d2, current_cost, indices, actions, idx_prob, action_prob)
        print(f"Iter {iter_}: best cost={current_cost} current cost={new_cost}")

        if new_cost > current_cost or random.random() < 0.5:
            current_cost = new_cost
            state = new_state


if __name__ == '__main__':
    with open('debug/test_refine.npy', 'rb') as afile:
        xyz_array = np.load(afile)
        xy_array = np.load(afile)
    print(xyz_array.shape)
    xyz_array2 = xyz_array.reshape((-1, 3))
    xy_array2 = xy_array.reshape((-1, 2))
    e = project_raw(xyz_array2, xy_array2)
    print(e)
    e = project_raw(xyz_array2, xy_array2)
    print(e)
    e = project_raw(xyz_array2, xy_array2)
    print(e)
    print(evaluate([0 for _ in range(xyz_array.shape[0])], xyz_array, xy_array))
    print(evaluate([1 for _ in range(xyz_array.shape[0])], xyz_array, xy_array))
    print(evaluate([2 for _ in range(xyz_array.shape[0])], xyz_array, xy_array))
    print(evaluate([3 for _ in range(xyz_array.shape[0])], xyz_array, xy_array))
    print(evaluate([4 for _ in range(xyz_array.shape[0])], xyz_array, xy_array))

    # simulated_annealing(xyz_array, xy_array)

