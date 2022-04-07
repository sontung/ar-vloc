import torch
import pycolmap
import colmap_db_read
import retrieval_based_no_mapper
import cv2
import sys
sys.path.append("Hierarchical-Localization")

from pathlib import Path
from hloc import extractors
from hloc import extract_features, match_features_bare
from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names
from hloc.triangulation import (import_features, import_matches, geometric_verification)
from hloc.reconstruction import create_empty_db, import_images, get_image_ids
from vis_utils import visualize_matching_pairs


def main():
    retrieval_dataset = Path('vloc_workspace_retrieval')
    retrieval_images_dir = retrieval_dataset / 'images_retrieval'
    retrieval_conf = extract_features.confs['sift']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_class = dynamic_load(extractors, retrieval_conf['model']['name'])
    retrieval_model = model_class(retrieval_conf['model']).eval().to(device)
    feature_path = Path(retrieval_dataset, retrieval_conf['output'] + '.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(list_h5_names(feature_path) if feature_path.exists() else ())

    database = retrieval_dataset / 'database_hloc.db'
    image_dir = retrieval_dataset / "images_retrieval"
    skip_geometric_verification = False
    system = retrieval_based_no_mapper.Localization()
    system.run_image_retrieval()

    extract_features.main_wo_model_loading(retrieval_model, device, skip_names,
                                           retrieval_conf, retrieval_images_dir,
                                           feature_path=feature_path)
    matching_conf = match_features_bare.confs["NN-ratio"]
    match_features_bare.main(matching_conf, retrieval_dataset/"pairs.txt", feature_path,
                             matches=retrieval_dataset/"matches.h5", overwrite=True)

    create_empty_db(database)
    import_images(image_dir, database, "PER_FOLDER", None)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, feature_path)
    import_matches(image_ids, database, retrieval_dataset/"pairs.txt", retrieval_dataset/"matches.h5",
                   None, skip_geometric_verification)
    if not skip_geometric_verification:
        geometric_verification(database, retrieval_dataset/"pairs.txt")

    system.read_matching_database(database)
    pairs, _ = system.read_2d_2d_matches("query/query.jpg", max_pool_size=50)

    best_score = None
    best_pose = None
    default_metadata = {'f': 26.0, 'cx': 1134.0, 'cy': 2016.0}

    for _ in range(10):
        r_mat, t_vec, score = retrieval_based_no_mapper.localize(default_metadata, pairs)
        if best_score is None or score > best_score:
            best_score = score
            best_pose = (r_mat, t_vec)
            if best_score == 1.0:
                break
    r_mat, t_vec = best_pose

    system.localization_results.append(((r_mat, t_vec), (1, 0, 0)))
    system.visualize()


def debug_matching():
    images_dir = "vloc_workspace_retrieval/images_retrieval"
    database_dir = "vloc_workspace_retrieval/database_hloc.db"
    matches, _ = colmap_db_read.extract_colmap_two_view_geometries(database_dir)
    id2kp, id2desc, id2name = colmap_db_read.extract_colmap_hloc(database_dir)
    for m in matches:
        img_id1, img_id2 = m
        print(id2name[img_id1], id2name[img_id2])
        img1 = cv2.imread(f"{images_dir}/{id2name[img_id1]}")
        img2 = cv2.imread(f"{images_dir}/{id2name[img_id2]}")
        pairs = []
        for fid1, fid2 in matches[m]:
            x1, y1 = id2kp[img_id1][fid1]
            x2, y2 = id2kp[img_id2][fid2]
            x1, y1 = map(int, (x1, y1))
            x2, y2 = map(int, (x2, y2))
            pair = ((x1, y1), (x2, y2))
            pairs.append(pair)
        img = visualize_matching_pairs(img1, img2, pairs)
        img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
        cv2.imshow("", img)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # main()
    debug_matching()
