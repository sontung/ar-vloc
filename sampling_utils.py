import numpy as np
from sklearn.cluster import MiniBatchKMeans


def random_select(pid_neighbors, pid_coord_list, nb_selection=10):
    nb_clusters = min(round(len(pid_neighbors)/nb_selection), 4)
    cluster_model = MiniBatchKMeans(nb_clusters, random_state=1)
    pose_arr = np.vstack(pid_coord_list)
    labels = cluster_model.fit_predict(pose_arr)
    cluster2idx = {v: [] for v in range(nb_clusters)}
    for u, v in enumerate(labels):
        cluster2idx[v].append(pid_neighbors[u])

    selection = []
    for _ in range(nb_selection):
        v = np.random.choice(list(range(nb_clusters)))
        points = cluster2idx[v]
        if len(points) < 1:
            continue
        u = np.random.choice(points)
        points.remove(u)
        cluster2idx[v] = points
        selection.append(u)
    return selection


def rank_by_area(cid2kp):
    total_area = 0
    for kp in cid2kp.values():
        total_area += len(kp)
    res = {}
    total_prob = 0
    for cid in cid2kp:
        prob = 1-len(cid2kp[cid])/total_area
        res[cid] = prob
        total_prob += prob
    for cid in cid2kp:
        old = res[cid]
        res[cid] = old/total_prob
    return res


def rank_by_response(cid2res):
    total_res = 0
    for res in cid2res.values():
        total_res += res
    result = {}
    for cid in cid2res:
        prob = cid2res[cid]/total_res
        result[cid] = prob
    return result


def combine(ranks):
    cid_list = ranks[0].keys()
    res = {cid: [] for cid in cid_list}
    for rank in ranks:
        for cid in rank:
            res[cid].append(rank[cid])
    for cid in res:
        res[cid] = np.mean(res[cid])
    return res