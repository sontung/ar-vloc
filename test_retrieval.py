import torch
import sys
sys.path.append("Hierarchical-Localization")

from pathlib import Path
from hloc import extract_features
from hloc import extractors
from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names


@profile
def main():
    dataset = Path('vloc_workspace_retrieval')  # change this if your dataset is somewhere else
    images_dir = dataset / 'images_retrieval'

    loc_pairs_dir = dataset / 'pairs.txt'  # top 20 retrieved by NetVLAD
    retrieval_conf = extract_features.confs['netvlad']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(extractors, retrieval_conf['model']['name'])
    model = Model(retrieval_conf['model']).eval().to(device)

    feature_path = None
    if feature_path is None:
        feature_path = Path(dataset, retrieval_conf['output']+'.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(list_h5_names(feature_path)
                     if feature_path.exists() else ())

    for _ in range(10):
        global_descriptors = extract_features.main_wo_model_loading(model, device, skip_names, retrieval_conf,
                                                                    images_dir, dataset)

    for _ in range(10):
        global_descriptors = extract_features.main(retrieval_conf, images_dir, dataset)


if __name__ == '__main__':
    main()