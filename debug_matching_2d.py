import sys
sys.path.append("d2-net")
from extract_features import extract_and_describe_using_d2_net
import cv2
import numpy as np
import imageio
from scipy.spatial import KDTree
from feature_matching import load_2d_queries_generic
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

im1 = imageio.imread("/home/sontung/work/ar-vloc/sfm_ws_hblab/images/IMG_0652.jpg")
kp1, score1, desc1 = extract_and_describe_using_d2_net(im1)
tree = KDTree(desc1)
descriptors, coordinates, im_names, md_list, im_list, response_list = load_2d_queries_generic(query_images_folder)
im2 = np.copy(im_list[0])
kp2, score2, desc2 = extract_and_describe_using_d2_net(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
indices = sorted(list(range(score2.shape[0])), key=lambda du: score2[du], reverse=True)

for ind in indices:
    im2_show = np.copy(im2)
    im1_show = np.copy(cv2.cvtColor(im1, cv2.COLOR_RGB2BGR))
    x, y = map(int, kp2[ind, :2])
    cv2.circle(im2_show, (x, y), 20, (128, 128, 0), -1)
    _, ind_list2 = tree.query(desc2[ind], 5)
    for du2, ind2 in enumerate(ind_list2):
        x2, y2 = map(int, kp1[ind2, :2])
        if du2 == 0:
            cv2.circle(im1_show, (x2, y2), 20, (0, 0, 255), 5)
        elif du2 == 1:
            cv2.circle(im1_show, (x2, y2), 20, (0, 255, 0), 5)
        elif du2 == 2:
            cv2.circle(im1_show, (x2, y2), 20, (255, 0, 0), 5)
        else:
            cv2.circle(im1_show, (x2, y2), 20, (0, 0, 0), 5)

    im_show = np.hstack([im2_show, im1_show])
    im_show = cv2.resize(im_show, (im_show.shape[1]//4, im_show.shape[0]//4))
    cv2.imshow("t", im_show)
    cv2.waitKey()
    cv2.destroyAllWindows()



# x, y = 372, 934
# _, ind = point2d_cloud.xy_tree.query([x, y], 1)
# fx, fy = point2d_cloud[ind].xy
# print(fx, fy)
#
# fx, fy = map(int, (fx // down_scale, fy // down_scale))
# cv2.circle(image, (fx, fy), 5, (128, 128, 0), 5)
# cv2.circle(image, (mouseX // down_scale, mouseY // down_scale), 5, (255, 0, 0), 5)
# cv2.imshow("image", image)
# # cv2.waitKey()
# point_ind, dist, _, ratio = point3d_cloud.matching_2d_to_3d_brute_force_no_ratio_test(point2d_cloud[ind].desc)
# images = visualize_matching_helper(np.copy(im_list[0]), point2d_cloud[ind],
#                                    point3d_cloud[point_ind], sfm_images_folder)
# cv2.imshow("t", images)
# cv2.waitKey()
# cv2.destroyWindow("t")
# sys.exit()

