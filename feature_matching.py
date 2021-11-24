import os

import matplotlib
matplotlib.use('TkAgg')
import random
import torch
from PIL import Image
import numpy as np
import kornia
import cv2
import pydegensac
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


def compute_kp_descriptors(img1):
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

    n_features = 4000
    detector = kornia.feature.ScaleSpaceDetector(n_features,
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
        descs = descriptor(patches.view(B * N, CH, H, W))
        # scores, matches = kornia.feature.match_snn(descs[0], descs[1], 0.9)

    # Now RANSAC
    src_pts = lafs[0, :, :, 2].data.cpu()

    img1 = np.array(img1)
    for i in range(src_pts.shape[0]):
        u, v = map(int, src_pts[i])
        cv2.circle(img1, (u, v), 1, (255, 128, 128), -1)
    return src_pts.unsqueeze(0), descs.unsqueeze(0)


if __name__ == '__main__':
    compute_kp_descriptors()