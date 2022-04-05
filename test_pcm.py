import torch
import sys
sys.path.append("Hierarchical-Localization")

from pathlib import Path
from hloc import extractors
from hloc import extract_features, match_features
from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names


retrieval_dataset = Path('vloc_workspace_retrieval')
retrieval_images_dir = retrieval_dataset / 'images_retrieval'
retrieval_conf = extract_features.confs['sift']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_class = dynamic_load(extractors, retrieval_conf['model']['name'])
retrieval_model = model_class(retrieval_conf['model']).eval().to(device)
feature_path = Path(retrieval_dataset, retrieval_conf['output'] + '.h5')
feature_path.parent.mkdir(exist_ok=True, parents=True)
skip_names = set(list_h5_names(feature_path) if feature_path.exists() else ())

extract_features.main_wo_model_loading(retrieval_model, device, skip_names,
                                       retrieval_conf, retrieval_images_dir,
                                       feature_path=feature_path)
matching_conf = match_features.confs["NN-ratio"]
match_features.main(matching_conf, retrieval_dataset/"pairs.txt", feature_path, matches=retrieval_dataset/"matches.h5")

# import_features(image_ids, database, features)
# import_matches(image_ids, database, pairs, matches,
#                min_match_score, skip_geometric_verification)
# if not skip_geometric_verification:
#     geometric_verification(database, pairs, verbose)