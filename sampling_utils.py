import numpy as np


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
    total_prob = 0
    for cid in cid2res:
        prob = cid2res[cid]/total_res
        result[cid] = prob
        total_prob += prob
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