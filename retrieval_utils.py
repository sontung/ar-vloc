import sys
from colmap_io import build_co_visibility_graph, read_images, read_name2id
from scipy.spatial import KDTree


class CandidatePool:
    def __init__(self):
        self.pool = []
        self.pid2votes = {}

    def __len__(self):
        return len(self.pool)

    def add(self, candidate):
        self.pool.append(candidate)
        pid = candidate.pid
        if pid not in self.pid2votes:
            self.pid2votes[pid] = 1
        else:
            self.pid2votes[pid] += 1

    def count_votes(self):
        # normalize candidates' distances
        max_dis = max([cand.dis for cand in self.pool])

        # normalize candidates' desc differences
        max_desc_diff = max([cand.desc_diff for cand in self.pool])

        # populate votes
        self.pid2votes = {}
        for candidate in self.pool:
            pid = candidate.pid
            dis = candidate.dis
            desc_diff = candidate.desc_diff
            if pid not in self.pid2votes:
                self.pid2votes[pid] = 0
            norm_dis = 1-dis/max_dis
            norm_desc_diff = 1-desc_diff/max_desc_diff
            vote = norm_dis + norm_desc_diff
            candidate.dis = norm_dis
            candidate.desc_diff = norm_desc_diff
            self.pid2votes[pid] += vote

    def filter(self):
        # filters pid
        pid2scores = {}
        for candidate in self.pool:
            pid = candidate.pid
            if pid not in pid2scores:
                pid2scores[pid] = [candidate]
            else:
                pid2scores[pid].append(candidate)

        new_pool = []
        for pid in pid2scores:
            candidates = pid2scores[pid]
            # best_candidate = min(candidates, key=lambda x: x.dis+x.desc_diff)
            best_candidate = max(candidates, key=lambda x: x.dis+x.desc_diff)
            new_pool.append(best_candidate)
        self.pool = new_pool

        # filters fid
        fid2scores = {}
        coord_array = [candidate.query_coord for candidate in self.pool]
        tree = KDTree(coord_array)
        for candidate in self.pool:
            dis, idx = tree.query(candidate.query_coord, 1)
            if dis == 0:
                key_ = f"{candidate.query_coord[0]}-{candidate.query_coord[1]}"
                if key_ not in fid2scores:
                    fid2scores[key_] = [candidate]
                else:
                    fid2scores[key_].append(candidate)

        new_pool = []
        for fid in fid2scores:
            candidates = fid2scores[fid]
            if len(candidates) > 1:
                best_candidate = max(candidates, key=lambda x: x.dis + x.desc_diff)
                new_pool.append(best_candidate)
            else:
                new_pool.extend(candidates)
        self.pool = new_pool

    def sort(self, by_votes=False):
        if by_votes:
            self.pool = sorted(self.pool, key=lambda x: self.pid2votes[x.pid], reverse=True)
        else:
            self.pool = sorted(self.pool, key=lambda x: x.dis)


class MatchCandidate:
    def __init__(self, query_coord, fid, pid, dis, desc_diff):
        self.query_coord = query_coord
        self.pid = pid
        self.fid = fid
        self.dis = dis
        self.desc_diff = desc_diff

    def __str__(self):
        return f"matched to {self.pid} with score={self.dis}"


def enhance_retrieval_pairs(dir_to_retrieval_pairs, image2pose, out_dir, extra_retrieval=20):
    """
    tries to add more retrieval pairs based on co-visibility of database images
    """
    sys.stdin = open(dir_to_retrieval_pairs, "r")
    lines = sys.stdin.readlines()
    query_name_to_database_names = {}
    for line in lines:
        line = line.rstrip()
        query_name, database_name = line.split(" ")
        if query_name not in query_name_to_database_names:
            query_name_to_database_names[query_name] = [database_name]
        else:
            query_name_to_database_names[query_name].append(database_name)
    image_id_to_visibilities, image_id_to_top_k = build_co_visibility_graph(image2pose)
    name2id = read_name2id(image2pose)
    for query_name in query_name_to_database_names:
        existing_names = query_name_to_database_names[query_name]
        existing_ids = [name2id[name] for name in existing_names]
        pool = []
        for database_id in existing_ids:
            top_k_database_ids = image_id_to_top_k[database_id]
            contribution = []
            for database_id2 in top_k_database_ids:
                if len(contribution) >= extra_retrieval*2:
                    break
                else:
                    if database_id2 not in existing_ids:
                        contribution.append((database_id2, image_id_to_visibilities[database_id][database_id2]))
            pool.extend(contribution)
        pool = sorted(pool, key=lambda du: du[1], reverse=True)
        final_contribution = []
        for database_id2, _ in pool:
            if len(final_contribution) >= extra_retrieval:
                break
            else:
                if database_id2 not in existing_ids and database_id2 not in final_contribution:
                    final_contribution.append(database_id2)
        existing_ids.extend(final_contribution)
        existing_names = [image2pose[img_id][0] for img_id in existing_ids]
        query_name_to_database_names[query_name] = existing_names

    # write results
    with open(out_dir, "w") as a_file:
        for query_name in query_name_to_database_names:
            existing_names = query_name_to_database_names[query_name]
            for name in existing_names:
                print(query_name, name, file=a_file)


if __name__ == '__main__':
    image2pose_ = read_images("/home/sontung/work/ar-vloc/vloc_workspace_retrieval/new/images.txt")
    enhance_retrieval_pairs("/home/sontung/work/ar-vloc/vloc_workspace_retrieval/retrieval_pairs.txt", image2pose_,
                            "/home/sontung/work/ar-vloc/vloc_workspace_retrieval/retrieval_pairs2.txt")