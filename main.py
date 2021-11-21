import numpy as np
import pickle
import os
import torch
import cv2
import kornia
from colmap_io import build_descriptors
from pathlib import Path
from scipy.spatial import KDTree
from matplotlib import pyplot as plt


def load_2d_queries(folder="test_images"):
    im_names_dir = "test_images/im_names.pkl"
    desc_dir = 'test_images/keypoint_descriptors.pt'

    my_file = Path(im_names_dir)
    if my_file.is_file():
        print("Loading 2D descriptors for test images at test_images/")
        descs = torch.load(desc_dir)
        with open("test_images/im_names.pkl", "rb") as fp:
            im_names = pickle.load(fp)
    else:
        im_names = os.listdir(folder)
        sift_model = kornia.feature.SIFTFeature(num_features=1000, device=torch.device("cpu"))
        descs = []
        for name in im_names:
            im_list = []
            im_name = os.path.join(folder, name)
            im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2))
            im_list.append(np.expand_dims(im, -1).astype(float))
            # cv2.imshow("t", im)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            an_im = kornia.utils.image_list_to_tensor(im_list)
            with torch.no_grad():
                laf, response, desc = sift_model.forward(an_im)
                descs.append(desc.cpu())
                points = kornia.feature.laf.get_laf_center(laf)
                px, py = kornia.feature.laf.get_laf_pts_to_draw(laf)
                plt.figure()
                plt.imshow(kornia.utils.tensor_to_image(an_im))
                plt.plot(px, py, 'r')
                plt.plot(points[0, :, 0], points[0, :, 1], 'bo')
                plt.show()
            break
        descs = torch.stack(descs)

        # torch.save(descs, desc_dir)
        # with open(im_names_dir, "wb") as fp:
        #     pickle.dump(im_names, fp)
        # print("Saved 2D descriptors at test_images/")

    return descs, im_names


def load_3d_database():
    point3d_ids_dir = "data/point3d_ids.pkl"
    point3d_descs_dir = "data/point3d_descs"

    my_file = Path(point3d_ids_dir)
    if my_file.is_file():
        print("Loading 3D descriptors at data/")
        point3d_desc_list = np.load(f"{point3d_descs_dir}.npy")
        with open(point3d_ids_dir, "rb") as fp:
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
        np.save(point3d_descs_dir, point3d_desc_list)
        with open(point3d_ids_dir, "wb") as fp:
            pickle.dump(point3d_id_list, fp)
    return point3d_id_list, point3d_desc_list


def matching_2d_to_3d(point3d_id_list, point3d_desc_list):
    kd_tree = KDTree(point3d_desc_list)
    for i in range(point3d_desc_list.shape[0]):
        desc = point3d_desc_list[i]
        res = kd_tree.query(desc, 2)
        # print(res[0][0], res[1][0])
        # break


def main():
    desc_list, im_name_list = load_2d_queries()
    point3d_id_list, point3d_desc_list = load_3d_database()
    matching_2d_to_3d(point3d_id_list, point3d_desc_list)


if __name__ == '__main__':
    main()
