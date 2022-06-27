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

SFM_FOLDER = "/media/sontung/580ECE740ECE4B28/7scenes_reference_models/7scenes_reference_models/office/sfm_gt"
SFM_IMAGES_FILE = f"{SFM_FOLDER}/images.txt"
WS_FOLDER = pathlib.Path("7scenes_ws_div")
QUERY_LIST = f"{SFM_FOLDER}/list_test.txt"
IMAGES_ROOT_FOLDER = pathlib.Path("/media/sontung/580ECE740ECE4B28/7_scenes_images/office")
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

@profile
def main(nb_neighbors=40):
    """
    will try to diversify the retrieval, avoid return too similar images
    """
    db_descriptors_dir = WS_FOLDER / "database_global_descriptors_0.pkl"
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
    query_image_names = ["seq-09/frame-000401.color.png"]

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

    collection2descmat = {}
    for name in database_image_names:
        collection_name = name.split("/")[0]
        if collection_name not in collection2descmat:
            collection2descmat[collection_name] = [[name2desc[name]], [name]]
        else:
            collection2descmat[collection_name][0].append(name2desc[name])
            collection2descmat[collection_name][1].append(name)

    desc_mat, all_names = collection2descmat["seq-03"]
    desc_mat = np.vstack(desc_mat).astype(np.float32)
    dim = db_mat.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(desc_mat)
    distances, indices = index.search(query_mat, nb_neighbors)
    left_out_list = []
    for idx22 in range(indices.shape[1] - 1):
        db_idx = indices[0][idx22]
        if all_names[db_idx] not in left_out_list:
            str_ = all_names[db_idx].split("/")[-1]
            str_ = str_.split(".color.png")[0].split("frame-")[-1]
            for i_ in range(-3, 4):
                if i_ != 0:
                    new_str = str(int(str_)+i_).zfill(len(str_))
                    im_name = f"seq-03/frame-{new_str}.color.png"
                    left_out_list.append(im_name)
            print(all_names[db_idx], distances[0][idx22] / distances[0][idx22 + 1])


if __name__ == '__main__':
    main()
