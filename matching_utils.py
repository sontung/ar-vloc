from scipy.spatial import KDTree
import cv2
import numpy as np


def match(desc_mat1, desc_mat2):
    tree2 = KDTree(desc_mat2)
    distances, indices = tree2.query(desc_mat1, 2)
    res = []
    for i in range(distances.shape[0]):
        if distances[i, 0] / distances[i, 1] < 0.8:
            res.append((i, indices[i, 0]))
    return res


def decolorize(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)


def draw_matches(kps1, kps2, tentatives, img1, img2, H, mask):
    if H is None:
        print("No homography found")
        return
    matchesMask = mask.ravel().tolist()
    h, w, ch = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    # print (dst)
    # Ground truth transformation
    img2_tr = cv2.polylines(decolorize(img2), [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
    # Blue is estimated, green is ground truth homography
    draw_params = dict(matchColor=(255, 255, 0),  # draw matches in yellow color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img_out = cv2.drawMatches(decolorize(img1), kps1, img2_tr, kps2, tentatives, **draw_params)
    return img_out