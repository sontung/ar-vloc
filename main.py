import numpy as np
import pickle
import os
import torch
from colmap_io import build_descriptors
from pathlib import Path
# from pykdtree.kdtree import KDTree
from scipy.spatial import KDTree


def load_3d_database():
    my_file = Path("data/point3d_ids.pkl")
    if my_file.is_file():
        print("Loading 3D descriptors at data/")
        point3d_desc_list = np.load("data/point3d_descs.npy")
        with open("data/point3d_ids.pkl", "rb") as fp:
            point3d_id_list = pickle.load(fp)
        print(f"\t{len(point3d_id_list)} 3D points with desc mat {point3d_desc_list.shape}")
    else:
        os.makedirs("data", exist_ok=True)
        _, point3did2descs = build_descriptors()
        point3d_id_list = []
        point3d_desc_list = []
        for point3d_id in point3did2descs:
            point3d_id_list.append(point3d_id)
            descs = [data[1] for data in point3did2descs[point3d_id]]
            mean_desc = torch.mean(torch.stack(descs), 0)
            point3d_desc_list.append(mean_desc.numpy())
        point3d_desc_list = np.vstack(point3d_desc_list)
        print("Saved 3D descriptors at data/")
        np.save("data/point3d_descs", point3d_desc_list)
        with open("data/point3d_ids.pkl", "wb") as fp:
            pickle.dump(point3d_id_list, fp)
    return point3d_id_list, point3d_desc_list


def matching_2d_to_3d(point3d_id_list, point3d_desc_list):
    kd_tree = KDTree(point3d_desc_list)
    for i in range(point3d_desc_list.shape[0]):
        desc = point3d_desc_list[i]
        res = kd_tree.query(desc, 2)
        print(res[0][0], res[1][0])
        # break


if __name__ == '__main__':
    p1, p2 = load_3d_database()
    matching_2d_to_3d(p1, p2)