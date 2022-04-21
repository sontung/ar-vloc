import cv2
import matching_utils

import numpy as np
import cv2
import pydegensac
from time import time
from copy import deepcopy
from vis_utils import concat_images_different_sizes, visualize_matching_pairs


img1 = cv2.imread('vloc_workspace_retrieval/images_retrieval/query/query.jpg')
img2 = cv2.imread('vloc_workspace_retrieval/images_retrieval/db/image0467.jpg')

detector = cv2.SIFT_create(edgeThreshold=10,
                           nOctaveLayers=4,
                           contrastThreshold=0.02, nfeatures=8000)
kps1, descs1 = detector.detectAndCompute(img1, None)
kps2, descs2 = detector.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(descs1, descs2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [False for i in range(len(matches))]

# SNN ratio test
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        matchesMask[i] = True
tentatives = [m[0] for i, m in enumerate(matches) if matchesMask[i]]

t = time()
src_pts = [kps1[m.queryIdx] for m in tentatives]
dst_pts = [kps2[m.trainIdx] for m in tentatives]

# mask = [True]*len(tentatives)
# H, mask = pydegensac.findHomography(src_pts,
#                                       dst_pts,
#                                       px_th = 4.0,
#                                       conf = 0.99,
#                                       max_iters = 2000,
#                                       laf_consistensy_coef=0,
#                                       error_type='sampson',
#                                       symmetric_error_check=True)

H_laf, mask = pydegensac.findHomography(src_pts,
                                      dst_pts,
                                      px_th=4.0,
                                      conf=0.99,
                                      max_iters=2000,
                                      laf_consistensy_coef=3.0,
                                      error_type='sampson',
                                      symmetric_error_check=True)

print('pydegensac found {} inliers'.format(int(deepcopy(mask).astype(np.float32).sum())))
pairs = []
for idx, pair in enumerate(tentatives):
    if mask[idx]:
        p1 = kps1[pair.queryIdx].pt
        p2 = kps2[pair.trainIdx].pt
        print(p1, p2)

        p1 = map(int, p1)
        p2 = map(int, p2)
        pairs.append((p1, p2))
img = visualize_matching_pairs(img1, img2, pairs)
img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))

cv2.imshow("", img)
cv2.waitKey()
cv2.destroyAllWindows()
