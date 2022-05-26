import os
import pathlib
import sys

sys.path.append("Hierarchical-Localization")
sys.path.append("cnnimageretrieval-pytorch")
import pickle
import faiss

import numpy as np

import colmap_io
from retrieval_utils import (extract_global_descriptors_on_database_images)

from torch.utils.model_zoo import load_url
from cirtorch.networks.imageretrievalnet import init_network

from pathlib import Path

from hloc import extractors
from hloc import extract_features, match_features_bare

from hloc.utils.base_model import dynamic_load
from hloc.triangulation import (import_features)
from hloc.reconstruction import create_empty_db, import_images, get_image_ids

SFM_FOLDER = "/media/sontung/580ECE740ECE4B28/7scenes_reference_models/7scenes_reference_models/redkitchen/sfm_gt"
SFM_IMAGES_FILE = f"{SFM_FOLDER}/images.txt"
WS_FOLDER = pathlib.Path("7scenes_ws_div")
QUERY_LIST = f"{SFM_FOLDER}/list_test.txt"
IMAGES_ROOT_FOLDER = pathlib.Path("/media/sontung/580ECE740ECE4B28/7_scenes_images/redkitchen")
DB_DIR = WS_FOLDER / "database.db"


def create_hloc_db(workspace_database_dir):
    image_list = []
    for subdir, dirs, files in os.walk(IMAGES_ROOT_FOLDER):
        for file in files:
            if "color" in file:
                full_path = os.path.join(subdir, file)
                path_ = full_path.split(str(IMAGES_ROOT_FOLDER) + "/")[-1]
                image_list.append(path_)

    if not workspace_database_dir.exists():
        create_empty_db(workspace_database_dir)
        import_images(IMAGES_ROOT_FOLDER, workspace_database_dir, "SINGLE", image_list)
    device = "cuda"
    matching_conf = extract_features.confs['sift']
    model_class = dynamic_load(extractors, matching_conf['model']['name'])
    matching_model = model_class(matching_conf['model']).eval().to(device)
    matching_feature_path = pathlib.Path(WS_FOLDER, matching_conf['output'] + '.h5')
    if not matching_feature_path.exists():
        extract_features.main_wo_model_loading_image_list(matching_model, device, image_list,
                                                          matching_conf, IMAGES_ROOT_FOLDER,
                                                          feature_path=matching_feature_path)
    image_ids = get_image_ids(workspace_database_dir)
    if not workspace_database_dir.exists():
        import_features(image_ids, workspace_database_dir, matching_feature_path)
    return matching_feature_path, image_list, image_ids


def copy_desc_mat(mat, name2desc, names):
    for u, name in enumerate(names):
        mat[u] = name2desc[name]


def extract_retrieval_pairs(db_descriptors_dir, query_image_names, database_image_names, save_file, nb_neighbors=40):
    with open(db_descriptors_dir, 'rb') as handle:
        database_descriptors = pickle.load(handle)
    img_names = database_descriptors["name"]
    img_descriptors = database_descriptors["desc"]
    name2desc = {}
    for u, name in enumerate(img_names):
        name2desc[name] = img_descriptors[u]
    query_mat = np.zeros((len(query_image_names), img_descriptors[0].shape[0]), dtype=np.float32)
    db_mat = np.zeros((len(database_image_names), img_descriptors[0].shape[0]), dtype=np.float32)
    copy_desc_mat(query_mat, name2desc, query_image_names)
    copy_desc_mat(db_mat, name2desc, database_image_names)

    dim = db_mat.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(db_mat)
    distances, indices = index.search(query_mat, nb_neighbors)
    with open(str(save_file), "w") as a_file:
        for u in range(len(query_image_names)):
            for v in range(nb_neighbors):
                db_idx = indices[u, v]
                print(query_image_names[u], database_image_names[db_idx], file=a_file)


def extract_retrieval_pairs_diversified(db_descriptors_dir, query_image_names,
                                        database_image_names, save_file, nb_neighbors=40, with_collection_info=True):
    with open(db_descriptors_dir, 'rb') as handle:
        database_descriptors = pickle.load(handle)
    img_names = database_descriptors["name"]
    img_descriptors = database_descriptors["desc"]
    name2desc = {}
    for u, name in enumerate(img_names):
        name2desc[name] = img_descriptors[u]
    query_mat = np.zeros((len(query_image_names), img_descriptors[0].shape[0]), dtype=np.float32)
    db_mat = np.zeros((len(database_image_names), img_descriptors[0].shape[0]), dtype=np.float32)
    copy_desc_mat(query_mat, name2desc, query_image_names)
    copy_desc_mat(db_mat, name2desc, database_image_names)

    if with_collection_info:
        collection2descmat = {}
        for name in database_image_names:
            collection_name = name.split("/")[0]
            if collection_name not in collection2descmat:
                collection2descmat[collection_name] = [[name2desc[name]], [name]]
            else:
                collection2descmat[collection_name][0].append(name2desc[name])
                collection2descmat[collection_name][1].append(name)

        with open(str(save_file), "w") as a_file:
            for collection_name in collection2descmat:
                desc_mat, all_names = collection2descmat[collection_name]
                desc_mat = np.vstack(desc_mat).astype(np.float32)

                dim = db_mat.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(desc_mat)
                distances, indices = index.search(query_mat, 10)
                ratio = distances[:, :-1] / distances[:, 1:]
                for i in range(distances.shape[0]):
                    accept = [0]
                    last_accept = True
                    for j in range(1, distances.shape[1]):
                        if ratio[i, j - 1] >= 0.99 and last_accept:
                            last_accept = False
                        else:
                            accept.append(j)
                            last_accept = True
                    for db_idx in indices[i, accept]:
                        print(query_image_names[i], all_names[db_idx], file=a_file)
    else:
        dim = db_mat.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(db_mat)
        distances, indices = index.search(query_mat, nb_neighbors)
        ratio = distances[:, :-1] / distances[:, 1:]

        with open(str(save_file), "w") as a_file:
            for i in range(distances.shape[0]):
                accept = [0]
                last_accept = True
                for j in range(1, distances.shape[1]):
                    if ratio[i, j - 1] >= 0.99 and last_accept:
                        last_accept = False
                    else:
                        accept.append(j)
                        last_accept = True
                for db_idx in indices[i, accept]:
                    print(query_image_names[i], database_image_names[db_idx], file=a_file)


def run_image_retrieval_and_matching(matching_feature_path, image_list, query_image_names, database_image_names,
                                     diversified_retrieval=False):
    retrieval_loc_pairs_dir = WS_FOLDER / 'pairs.txt'  # top 20 retrieved by NetVLAD
    retrieval_conf = extract_features.confs['netvlad']
    feature_path = Path(WS_FOLDER, retrieval_conf['output'] + '.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)

    # cnn image retrieval
    training_weights = 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth'
    weights_folder = "cnnimageretrieval-pytorch/data/networks"
    state = load_url(training_weights, model_dir=weights_folder)
    cnn_retrieval_net = init_network({'architecture': state['meta']['architecture'],
                                      'pooling': state['meta']['pooling'],
                                      'whitening': state['meta'].get('whitening')})
    cnn_retrieval_net.load_state_dict(state['state_dict'])
    cnn_retrieval_net.eval()
    cnn_retrieval_net.cuda()
    db_descriptors_dir = WS_FOLDER / "database_global_descriptors_0.pkl"
    if not db_descriptors_dir.exists():
        print("Global database descriptors not found, extracting ...")
        extract_global_descriptors_on_database_images(IMAGES_ROOT_FOLDER, WS_FOLDER,
                                                      multi_scale=False, image_list=image_list)

    # retrieve
    if not diversified_retrieval:
        extract_retrieval_pairs(db_descriptors_dir, query_image_names, database_image_names, retrieval_loc_pairs_dir)
    else:
        extract_retrieval_pairs_diversified(db_descriptors_dir, query_image_names,
                                            database_image_names, retrieval_loc_pairs_dir)

    # match
    matching_results_dir = WS_FOLDER / "matches.h5"
    if not matching_results_dir.exists():
        matching_conf = match_features_bare.confs["NN-ratio"]
        name2ref = match_features_bare.return_name2ref(matching_feature_path)
        match_features_bare.main_evaluation(name2ref, matching_conf, retrieval_loc_pairs_dir, matching_feature_path,
                                            matches=matching_results_dir, overwrite=True)


def prepare():
    my_file = pathlib.Path(f"{WS_FOLDER}/image2pose.pkl")
    if my_file.is_file():
        with open(f"{WS_FOLDER}/image2pose.pkl", 'rb') as handle:
            image2pose_ = pickle.load(handle)
    else:
        image2pose_ = colmap_io.read_images(SFM_IMAGES_FILE, by_im_name=True)
        with open(f"{WS_FOLDER}/image2pose.pkl", 'wb') as handle:
            pickle.dump(image2pose_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    database_image_names = []
    query_image_names = colmap_io.read_image_list(QUERY_LIST)
    keys = list(image2pose_.keys())
    for img in keys:
        if img not in query_image_names:
            database_image_names.append(img)
        else:
            del image2pose_[img]

    matching_feature_path, image_list, name2id = create_hloc_db(DB_DIR)
    run_image_retrieval_and_matching(matching_feature_path, image_list, query_image_names, database_image_names,
                                     diversified_retrieval=True)
    return query_image_names, database_image_names


def read_logs(log_file):
    sys.stdin = open(log_file, "r")
    records = []
    for line in sys.stdin.readlines():
        line = line[:-1]
        name, err = line.split(" ")
        err = float(err)
        records.append((name, err))
    records = sorted(records, key=lambda du: du[-1], reverse=True)
    return records


