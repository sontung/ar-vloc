import os

import matplotlib
matplotlib.use('TkAgg')
import random
import torch
from PIL import Image
from colmap_io import read_images
import numpy as np
import kornia
import cv2
import pydegensac
from matplotlib import pyplot as plt
from scipy.spatial import KDTree


def load_2d_queries_opencv(folder="test_images"):
    im_names = os.listdir(folder)
    descriptors = []
    coordinates = []
    for name in im_names:
        im_name = os.path.join(folder, name)
        im = cv2.imread(im_name)
        coord, desc = compute_kp_descriptors_opencv(im)
        coord = np.array(coord)
        coordinates.append(coord)
        descriptors.append(desc)
    return descriptors, coordinates, im_names


def build_descriptors_2d():
    images = read_images()
    point3did2descs = {}
    imageid2point3did2feature = {image_id: {} for image_id in images}
    matching_ratio = []
    for image_id in images:
        image_name = images[image_id][0]
        image_name = f"sfm_models/images/{image_name}"
        im = cv2.imread(image_name)
        coord, desc = compute_kp_descriptors_opencv(im)

        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # coord, desc = compute_kp_descriptors(im, n_features=8000)
        # coord = coord.squeeze().numpy()

        tree = KDTree(coord)
        total_dis = 0
        nb_points = 0
        nb_3d_points = 0
        points2d_meaningful = images[image_id][1]

        for x, y, p3d_id in points2d_meaningful:
            if p3d_id > 0:
                dis, idx = tree.query([x, y], 1)
                nb_3d_points += 1
                if dis < 8:
                    total_dis += dis
                    nb_points += 1
                    if p3d_id not in point3did2descs:
                        point3did2descs[p3d_id] = [[image_id, desc[idx]]]
                    else:
                        point3did2descs[p3d_id].append([image_id, desc[idx]])

        matching_ratio.append(nb_points/nb_3d_points)
    print(f"{np.mean(matching_ratio)*100}% of 3D points found descriptors")
    p3d_id_list = []
    p3d_desc_list = []
    for p3d_id in point3did2descs:
        p3d_id_list.append(p3d_id)
        desc_list = [du[1] for du in point3did2descs[p3d_id]]
        desc = np.mean(desc_list, axis=0)
        p3d_desc_list.append(desc)
    return p3d_id_list, p3d_desc_list


def compute_kp_descriptors_opencv(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp_list, des = sift.detectAndCompute(img, None)
    coords = []
    for kp in kp_list:
        coords.append(list(kp.pt))
    return coords, des


def compute_kp_descriptors(img1, n_features=4000):
    # img1 = Image.open(f'/home/sontung/work/ar-vloc/test_images/252824182_4725834497437928_3707047345468087757_n.jpg')
    # img1 = img1.resize((img1.size[0] // 4, img1.size[1] // 4))

    timg = kornia.utils.image_to_tensor(img1, False).float() / 255.
    timg_gray = kornia.color.rgb_to_grayscale(timg)

    # Now lets define  affine local deature detector and descriptor

    device = torch.device('cpu')

    PS = 41

    sift = kornia.feature.SIFTDescriptor(PS, rootsift=True).to(device)
    descriptor = sift

    resp = kornia.feature.BlobHessian()
    scale_pyr = kornia.geometry.ScalePyramid(3, 1.6, PS, double_image=True)

    nms = kornia.geometry.ConvQuadInterp3d(10)

    detector = kornia.feature.ScaleSpaceDetector(n_features,
                                                 resp_module=resp,
                                                 nms_module=nms,
                                                 scale_pyr_module=scale_pyr,
                                                 ori_module=kornia.feature.LAFOrienter(32),
                                                 aff_module=kornia.feature.LAFAffineShapeEstimator(32),
                                                 mr_size=6.0).to(device)

    with torch.no_grad():
        lafs, resps = detector(timg_gray)
        patches = kornia.feature.extract_patches_from_pyramid(timg_gray, lafs, PS)
        B, N, CH, H, W = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :)
        descs = descriptor(patches.view(B * N, CH, H, W))
        # scores, matches = kornia.feature.match_snn(descs[0], descs[1], 0.9)

    # Now RANSAC
    src_pts = lafs[0, :, :, 2].data.cpu()

    # img1 = np.array(img1)
    # for i in range(src_pts.shape[0]):
    #     u, v = map(int, src_pts[i])
    #     cv2.circle(img1, (u, v), 1, (255, 128, 128), -1)
    return src_pts.unsqueeze(0), descs.unsqueeze(0)


if __name__ == '__main__':
    build_descriptors_2d()