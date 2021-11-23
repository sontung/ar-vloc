import os

import matplotlib
matplotlib.use('TkAgg')
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import kornia
import cv2
import pydegensac
from kornia.feature import *
from time import time
import torch.optim as optim
from torch.nn import Parameter
from kornia.color import rgb_to_grayscale
from matplotlib import pyplot as plt


def visualize_LAF(img, LAF, img_idx = 0):
    x, y = kornia.feature.laf.get_laf_pts_to_draw(LAF, img_idx)
    plt.figure()
    plt.imshow(kornia.utils.tensor_to_image(img[img_idx]))
    plt.plot(x, y, 'r')
    plt.show()
    return


def draw_matches(im1, im2, mask, x1, x2):
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
    im = np.hstack([im1, im2])

    for i in range(x1.shape[0]):
        if mask[i]:
            u1, v1 = x1[i]
            u2, v2 = x2[i]
            color = [random.random()*255 for _ in range(3)]
            cv2.circle(im, (int(u1), int(v1)), 5, color, -1)
            cv2.circle(im, (int(u2+im1.shape[1]), int(v2)), 5, color, -1)
            cv2.line(im, (int(u1), int(v1)), (int(u2+im1.shape[1]), int(v2)), color)
    cv2.imshow("", im)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return


def main():
    files = [name for name in os.listdir('/home/sontung/work/ar-vloc/test_images/') if ".jpg" in name]
    files2 = [name for name in os.listdir('/home/sontung/work/ar-vloc/sfm_models/images/') if ".jpg" in name]

    img1 = Image.open(f'/home/sontung/work/ar-vloc/test_images/{random.choice(files)}')
    img1 = img1.resize((img1.size[0] // 4, img1.size[1] // 4))
    img2 = Image.open(
        f'/home/sontung/work/ar-vloc/sfm_models/images/{random.choice(files2)}').resize(
        img1.size)

    timg1 = kornia.utils.image_to_tensor(np.array(img1), False).float() / 255.
    timg2 = kornia.utils.image_to_tensor(np.array(img2), False).float() / 255.

    timg = torch.cat([timg1, timg2], dim=0)

    timg_gray = kornia.color.rgb_to_grayscale(timg)

    # Now lets define  affine local deature detector and descriptor

    device = torch.device('cpu')
    # device = torch.device('cuda:0')

    PS = 41

    sift = kornia.feature.SIFTDescriptor(PS, rootsift=True).to(device)
    descriptor = sift

    resp = BlobHessian()
    scale_pyr = kornia.geometry.ScalePyramid(3, 1.6, PS, double_image=True)

    nms = kornia.geometry.ConvQuadInterp3d(10)

    n_features = 4000
    detector = ScaleSpaceDetector(n_features,
                                  resp_module=resp,
                                  nms_module=nms,
                                  scale_pyr_module=scale_pyr,
                                  ori_module=kornia.feature.LAFOrienter(19),
                                  aff_module=kornia.feature.LAFAffineShapeEstimator(19),
                                  mr_size=6.0).to(device)

    with torch.no_grad():
        lafs, resps = detector(timg_gray)
        patches = kornia.feature.extract_patches_from_pyramid(timg_gray, lafs, PS)
        B, N, CH, H, W = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit :)
        descs = descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
        scores, matches = kornia.feature.match_snn(descs[0], descs[1], 0.9)

    # Now RANSAC
    src_pts = lafs[0, matches[:, 0], :, 2].data.cpu().numpy()
    dst_pts = lafs[1, matches[:, 1], :, 2].data.cpu().numpy()

    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0, 0.999, 10000)
    F, mask = pydegensac.findHomography(src_pts, dst_pts, 0.75, 0.99, 100000)

    inliers = matches[torch.from_numpy(mask).bool().squeeze(), :]
    draw_matches(np.array(img1), np.array(img2), mask, src_pts, dst_pts)
    print(len(inliers), 'inliers')

for _ in range(10):
    main()