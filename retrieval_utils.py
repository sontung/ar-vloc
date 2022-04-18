import sys
sys.path.append("cnnimageretrieval-pytorch")
import cv2
import os
import numpy as np
import pickle
import faiss
from colmap_io import build_co_visibility_graph, read_images, read_name2id
from vis_utils import concat_images_different_sizes, visualize_matching_pairs
from scipy.spatial import KDTree
from tqdm import tqdm

from os.path import isfile, join
from torch.utils.model_zoo import load_url
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_ms, extract_ss
from cirtorch.datasets.datahelpers import imresize, default_loader
from cirtorch.utils.general import get_data_root
from cirtorch.utils.whiten import whitenapply


TRAINED = {
    'rSfM120k-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'retrievalSfM120k-resnet101-gem': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth'
}


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


def extract_global_descriptors_on_database_images(database_folder, save_folder, multi_scale=True):
    all_images = [f for f in os.listdir(database_folder) if isfile(join(database_folder, f))]

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    input_res = 1024
    scales = [1, 1 / np.sqrt(2), 1 / 2]  # re-scaling factors for multi-scale extraction

    # sample image
    state = load_url(TRAINED['retrievalSfM120k-resnet101-gem'], model_dir=os.path.join(get_data_root(), 'networks'))
    net = init_network({'architecture': state['meta']['architecture'], 'pooling': state['meta']['pooling'],
                        'whitening': state['meta'].get('whitening')})
    net.load_state_dict(state['state_dict'])
    net.eval()
    net.cuda()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=state['meta']['mean'], std=state['meta']['std'])])
    names = []
    descriptors = []
    for img_file in tqdm(all_images):
        img = default_loader(join(database_folder, img_file))

        if not multi_scale:
            # single-scale extraction
            vec = extract_ss(net, transform(imresize(img, input_res)).unsqueeze(0).cuda())
            vec = vec.data.cpu().numpy()
            whiten_ss = state['meta']['Lw']['retrieval-SfM-120k']['ss']
            vec = whitenapply(vec.reshape(-1, 1), whiten_ss['m'], whiten_ss['P']).reshape(-1)
        else:
            # multi-scale extraction
            vec = extract_ms(net, transform(imresize(img, input_res)).unsqueeze(0).cuda(), ms=scales, msp=net.pool.p.item())
            vec = vec.data.cpu().numpy()
            whiten_ms = state['meta']['Lw']['retrieval-SfM-120k']['ms']
            vec = whitenapply(vec.reshape(-1, 1), whiten_ms['m'], whiten_ms['P']).reshape(-1)

        names.append(img_file)
        descriptors.append(vec)

    data = {"name": names, "desc": descriptors}
    with open(f'{save_folder}/database_global_descriptors_{int(multi_scale)}.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def extract_retrieval_pairs(database_descriptors_file, query_im_file, output_file_dir,
                            multi_scale=True, nb_neighbors=40, debug=False):
    with open(database_descriptors_file, 'rb') as handle:
        database_descriptors = pickle.load(handle)
    img_names = database_descriptors["name"]
    img_descriptors = database_descriptors["desc"]

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    input_res = 1024
    scales = [1, 1 / np.sqrt(2), 1 / 2]  # re-scaling factors for multi-scale extraction

    # sample image
    state = load_url(TRAINED['retrievalSfM120k-resnet101-gem'], model_dir=os.path.join(get_data_root(), 'networks'))
    net = init_network({'architecture': state['meta']['architecture'], 'pooling': state['meta']['pooling'],
                        'whitening': state['meta'].get('whitening')})
    net.load_state_dict(state['state_dict'])
    net.eval()
    net.cuda()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=state['meta']['mean'], std=state['meta']['std'])])

    img = default_loader(query_im_file)

    if not multi_scale:
        # single-scale extraction
        vec = extract_ss(net, transform(imresize(img, input_res)).unsqueeze(0).cuda())
        vec = vec.data.cpu().numpy()
        whiten_ss = state['meta']['Lw']['retrieval-SfM-120k']['ss']
        vec = whitenapply(vec.reshape(-1, 1), whiten_ss['m'], whiten_ss['P']).reshape(-1)
    else:
        # multi-scale extraction
        vec = extract_ms(net, transform(imresize(img, input_res)).unsqueeze(0).cuda(), ms=scales,
                         msp=net.pool.p.item())
        vec = vec.data.cpu().numpy()
        whiten_ms = state['meta']['Lw']['retrieval-SfM-120k']['ms']
        vec = whitenapply(vec.reshape(-1, 1), whiten_ms['m'], whiten_ms['P']).reshape(-1)
    img_descriptors = np.vstack(img_descriptors).astype(np.float32)
    vec = np.vstack([vec]).astype(np.float32)

    dim = img_descriptors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(img_descriptors)
    distances, indices = index.search(vec, nb_neighbors)
    if debug:
        db_img_root = "/home/sontung/work/ar-vloc/vloc_workspace_retrieval/images_retrieval/db"
        query_im = cv2.imread(query_im_file)
        for ind2, ind in enumerate(indices[0]):
            print(distances[0][ind2], img_names[ind])
            img = cv2.imread(f"{db_img_root}/{img_names[ind]}")
            img = concat_images_different_sizes([img, query_im])
            img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
            cv2.imshow("", img)
            cv2.waitKey()
            cv2.destroyAllWindows()
    with open(output_file_dir, "w") as a_file:
        for ind in indices[0]:
            print(f"query.jpg {img_names[ind]}", file=a_file)


def log_matching(pairs, name1, name2, name3):
    db_img = cv2.imread(name1)
    query_img = cv2.imread(name2)
    img = visualize_matching_pairs(query_img, db_img, pairs)
    cv2.imwrite(name3, img)


if __name__ == '__main__':
    # extract_global_descriptors_on_database_images("vloc_workspace_retrieval/images_retrieval/db",
    #                                               "vloc_workspace_retrieval")
    # extract_global_descriptors_on_database_images("vloc_workspace_retrieval/images_retrieval/db",
    #                                               "vloc_workspace_retrieval",
    #                                               multi_scale=False)
    extract_retrieval_pairs("vloc_workspace_retrieval/database_global_descriptors_0.pkl",
                            "vloc_workspace_retrieval/images_retrieval/query/query.jpg",
                            "vloc_workspace_retrieval/retrieval_pairs.txt",
                            multi_scale=False,
                            nb_neighbors=40)
