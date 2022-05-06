import pathlib
import pickle
import sys

import colmap_io

sys.path.append("Hierarchical-Localization")
sys.path.append("cnnimageretrieval-pytorch")

from pathlib import Path
from feature_matching import run_d2_detector_on_folder

from hloc import extractors
from hloc import extract_features, pairs_from_retrieval, match_features_bare

from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names, get_matches_wo_loading
from hloc.triangulation import (import_features)
from hloc.reconstruction import create_empty_db, import_images, get_image_ids

SFM_FOLDER = "/media/sontung/580ECE740ECE4B28/7scenes_reference_models/7scenes_reference_models/redkitchen/sfm_gt"
SFM_IMAGES_FILE = f"{SFM_FOLDER}/images.txt"
WS_FOLDER = pathlib.Path("7scenes_ws")
QUERY_LIST = f"{SFM_FOLDER}/list_test.txt"
IMAGES_ROOT_FOLDER = pathlib.Path("/media/sontung/580ECE740ECE4B28/7_scenes_images/redkitchen")
DB_DIR = WS_FOLDER / "database.db"


def create_hloc_db(workspace_database_dir):
    if not workspace_database_dir.exists():
        create_empty_db(workspace_database_dir)
        import_images(IMAGES_ROOT_FOLDER, workspace_database_dir, "SINGLE", None)
    device = "cuda"
    matching_conf = extract_features.confs['sift']
    model_class = dynamic_load(extractors, matching_conf['model']['name'])
    matching_model = model_class(matching_conf['model']).eval().to(device)
    matching_feature_path = pathlib.Path(WS_FOLDER, matching_conf['output'] + '.h5')
    extract_features.main_wo_model_loading(matching_model, device, [],
                                           matching_conf, IMAGES_ROOT_FOLDER,
                                           feature_path=matching_feature_path)
        # image_ids = get_image_ids(workspace_database_dir)
        # database_image_ids = {u: v for u, v in image_ids.items() if "db" in u}
        # import_features(database_image_ids, workspace_database_dir, matching_feature_path)


def main():
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

    create_hloc_db(DB_DIR)


if __name__ == '__main__':
    main()
