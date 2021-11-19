import numpy as np
import pickle
import os
import torch
import cv2
import kornia
from colmap_io import build_descriptors
from pathlib import Path
# from pykdtree.kdtree import KDTree
from scipy.spatial import KDTree


def load_2d_queries(folder="test_images"):
    im_names = os.listdir(folder)
    sift_model = kornia.feature.SIFTFeature(num_features=1000)
    im_list = []
    for name in im_names:
        im_name = os.path.join(folder, name)
        im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (1536//2, 2048//2))
        im_list.append(np.expand_dims(im, -1).astype(float))
        # cv2.imshow("t", im)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    im_list = kornia.utils.image_list_to_tensor(im_list)
    for i in range(im_list.size(0)):
        an_im = torch.unsqueeze(im_list[i], 0)

        with torch.no_grad():
            _, _, descs = sift_model.forward(an_im)


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
        # print(res[0][0], res[1][0])
        # break


if __name__ == '__main__':
    load_2d_queries()
    p1, p2 = load_3d_database()
    matching_2d_to_3d(p1, p2)
