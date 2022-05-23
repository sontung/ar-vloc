import pathlib
import pickle
import faiss
import numpy as np
import sys
import colmap_io

SFM_FOLDER = "/media/sontung/580ECE740ECE4B28/7scenes_reference_models/7scenes_reference_models/redkitchen/sfm_gt"
SFM_IMAGES_FILE = f"{SFM_FOLDER}/images.txt"
WS_FOLDER = pathlib.Path("7scenes_ws")
QUERY_LIST = f"{SFM_FOLDER}/list_test.txt"
IMAGES_ROOT_FOLDER = pathlib.Path("/media/sontung/580ECE740ECE4B28/7_scenes_images/redkitchen")
DB_DIR = WS_FOLDER / "database.db"


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
    ratio = distances[:, :-1]/distances[:, 1:]

    for i in range(distances.shape[0]):
        accept = [0]
        last_accept = True
        for j in range(1, nb_neighbors):
            if ratio[i, j-1] >= 0.99 and last_accept:
                last_accept = False
            else:
                accept.append(j)
                last_accept = True
        print(indices[i].shape, indices[i, accept].shape)


def main():
    my_file = pathlib.Path(f"{WS_FOLDER}/image2pose.pkl")
    if my_file.is_file():
        with open(f"{WS_FOLDER}/image2pose.pkl", 'rb') as handle:
            image2pose_ = pickle.load(handle)
    else:
        image2pose_ = colmap_io.read_images(SFM_IMAGES_FILE, by_im_name=True)
        with open(f"{WS_FOLDER}/image2pose.pkl", 'wb') as handle:
            pickle.dump(image2pose_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    db_descriptors_dir = WS_FOLDER / "database_global_descriptors_0.pkl"
    query_image_names = colmap_io.read_image_list(QUERY_LIST)
    database_image_names = []
    keys = list(image2pose_.keys())
    for img in keys:
        if img not in query_image_names:
            database_image_names.append(img)
        else:
            del image2pose_[img]
    retrieval_loc_pairs_dir = WS_FOLDER / 'pairs.txt'  # top 20 retrieved by NetVLAD
    extract_retrieval_pairs(db_descriptors_dir, query_image_names, database_image_names, retrieval_loc_pairs_dir)


if __name__ == '__main__':
    main()