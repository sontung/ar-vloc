import sys
sys.path.append("d2-net")
from extract_features import extract_and_describe_using_d2_net
import cv2
import random
import numpy as np
import imageio
from scipy.spatial import KDTree
from tqdm import tqdm
from feature_matching import load_2d_queries_generic, compute_kp_descriptors_opencv_with_d2_detector
from point2d import FeatureCloud
from point3d import PointCloud
from colmap_io import read_points3D_coordinates, read_images
from feature_matching import build_descriptors_2d
from vocab_tree import VocabTree
from vis_utils import visualize_matching, visualize_matching_helper


query_images_folder = "Test line small"
sfm_images_dir = "sfm_ws_hblab/images.txt"
sfm_point_cloud_dir = "sfm_ws_hblab/points3D.txt"
sfm_images_folder = "sfm_ws_hblab/images"


def sift_matching(_img1, _img2):
    _img1 = cv2.cvtColor(_img1, cv2.COLOR_RGB2BGR)

    _img1_show = np.copy(_img1)
    _img2_show = np.copy(_img2)
    im_show = np.hstack([_img1_show, _img2_show])

    _img1 = cv2.cvtColor(_img1, cv2.COLOR_BGR2GRAY)
    _img2 = cv2.cvtColor(_img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(edgeThreshold=10,
                           nOctaveLayers=4,
                           contrastThreshold=0.02, nfeatures=8000)
    _kp_list1, _des1 = sift.detectAndCompute(_img1, None)
    _des1 /= (np.linalg.norm(_des1, axis=1, ord=2, keepdims=True) + 1e-7)
    _des1 /= (np.linalg.norm(_des1, axis=1, ord=1, keepdims=True) + 1e-7)
    _des1 = np.sqrt(_des1)

    _kp_list2, _des2 = sift.detectAndCompute(_img2, None)
    _des2 /= (np.linalg.norm(_des2, axis=1, ord=2, keepdims=True) + 1e-7)
    _des2 /= (np.linalg.norm(_des2, axis=1, ord=1, keepdims=True) + 1e-7)
    _des2 = np.sqrt(_des2)

    _tree = KDTree(_des2)
    for _idx, _kp in enumerate(tqdm(_kp_list1)):
        dis, ind_list2 = _tree.query(_des1[_idx], 2)
        if dis[0] < dis[1]*0.7:
            color = (random.random()*255, random.random()*255, random.random()*255)
            x1, y1 = map(int, list(_kp.pt))
            x2, y2 = map(int, list(_kp_list2[ind_list2[0]].pt))
            cv2.circle(im_show, (x1, y1), 20, color, -1)
            cv2.circle(im_show, (x2+_img2_show.shape[1], y2), 20, color, -1)
            cv2.line(im_show, (x1, y1), (x2+_img2_show.shape[1], y2), color, 5)
    im_show = cv2.resize(im_show, (im_show.shape[1] // 4, im_show.shape[0] // 4))
    cv2.imshow("t", im_show)
    cv2.waitKey()
    cv2.destroyAllWindows()


def debug_one_by_one():
    im1 = imageio.imread("/home/sontung/work/ar-vloc/sfm_ws_hblab/images/IMG_0722.jpg")
    # kp1, desc1, score1 = extract_and_describe_using_d2_net(im1)
    kp1, desc1, score1 = compute_kp_descriptors_opencv_with_d2_detector(cv2.cvtColor(im1, cv2.COLOR_RGB2BGR),
                                                                        return_response=True)
    kp1 = np.array(kp1)

    tree = KDTree(desc1)
    descriptors, coordinates, im_names, md_list, im_list, response_list = load_2d_queries_generic(query_images_folder)
    im2 = np.copy(im_list[0])
    # kp2, desc2, score2 = extract_and_describe_using_d2_net(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
    kp2, desc2, score2 = compute_kp_descriptors_opencv_with_d2_detector(im2, return_response=True)
    kp2 = np.array(kp2)

    indices = sorted(list(range(len(score2))), key=lambda du: score2[du], reverse=True)

    for ind in indices:
        im2_show = np.copy(im2)
        im1_show = np.copy(cv2.cvtColor(im1, cv2.COLOR_RGB2BGR))
        x, y = map(int, kp2[ind, :2])
        cv2.circle(im2_show, (x, y), 20, (128, 128, 0), -1)
        _, ind_list2 = tree.query(desc2[ind], 10)
        for du2, ind2 in enumerate(ind_list2):
            x2, y2 = map(int, kp1[ind2, :2])
            if du2 == 0:
                cv2.circle(im1_show, (x2, y2), 20, (0, 0, 255), -1)
            elif du2 == 1:
                cv2.circle(im1_show, (x2, y2), 20, (0, 255, 0), -1)
            elif du2 == 2:
                cv2.circle(im1_show, (x2, y2), 20, (255, 0, 0), -1)
            else:
                cv2.circle(im1_show, (x2, y2), 20, (0, 0, 0), 5)

        im_show = np.hstack([im2_show, im1_show])
        im_show = cv2.resize(im_show, (im_show.shape[1]//4, im_show.shape[0]//4))
        cv2.imshow("t", im_show)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    im1 = imageio.imread("/home/sontung/work/ar-vloc/sfm_ws_hblab/images/IMG_0722.jpg")
    descriptors, coordinates, im_names, md_list, im_list, response_list = load_2d_queries_generic(query_images_folder)
    im2 = np.copy(im_list[0])
    sift_matching(im1, im2)
