import cv2
import numpy as np
from feature_matching import load_2d_queries_generic
from point2d import FeatureCloud
from point3d import PointCloud
from colmap_io import read_points3D_coordinates, read_images
from feature_matching import build_descriptors_2d
from vocab_tree import VocabTree
from vis_utils import visualize_matching, visualize_matching_helper


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY, mouseX2, mouseY2, down_scale
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(image, (x, y), 5, (255, 0, 0), 5)
        mouseX, mouseY = x*down_scale, y*down_scale


query_images_folder = "Test line small"
sfm_images_dir = "sfm_ws_hblab/images.txt"
sfm_point_cloud_dir = "sfm_ws_hblab/points3D.txt"
sfm_images_folder = "sfm_ws_hblab/images"
image2pose = read_images(sfm_images_dir)
point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)
point3d_id_list, point3d_desc_list, p3d_desc_list_multiple, point3did2descs = build_descriptors_2d(image2pose, sfm_images_folder)

point3d_cloud = PointCloud(point3did2descs, debug=True)
for i in range(len(point3d_id_list)):
    point3d_id = point3d_id_list[i]
    point3d_desc = point3d_desc_list[i]
    xyzrgb = point3did2xyzrgb[point3d_id]
    point3d_cloud.add_point(point3d_id, point3d_desc, p3d_desc_list_multiple[i], xyzrgb[:3], xyzrgb[3:])

point3d_cloud.commit(image2pose)
vocab_tree = VocabTree(point3d_cloud)
descriptors, coordinates, im_names, md_list, im_list, response_list = load_2d_queries_generic(query_images_folder)
point2d_cloud = FeatureCloud()

for j in range(coordinates[0].shape[0]):
    point2d_cloud.add_point(j, descriptors[0][j], coordinates[0][j], response_list[0][j])
point2d_cloud.nearby_feature(0, 2)

image_ori = im_list[0]
down_scale = 5
image_ori = cv2.resize(image_ori, (image_ori.shape[1]//down_scale, image_ori.shape[0]//down_scale))

mouseX = 2032
mouseY = 440
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
image = image_ori.copy()
MODE = None
RESULT = {}
RESULT_NO_RATIO = {}
INTERACTIVE = True

if INTERACTIVE:
    while True:
        cv2.imshow("image", image)

        if MODE == 1:
            if mouseX < 0:
                continue
            if (mouseX, mouseY) not in RESULT:
                _, ind = point2d_cloud.xy_tree.query([mouseX, mouseY], 1)
                point_ind, dist, _ = point3d_cloud.matching_2d_to_3d_brute_force(point2d_cloud[ind].desc,
                                                                                 returning_index=True)
                RESULT[(mouseX, mouseY)] = (ind, point_ind, dist)
                print(f"ratio test: {mouseX, mouseY} => {ind, point_ind, dist}")
            else:
                ind, point_ind, dist = RESULT[(mouseX, mouseY)]
            if point_ind is not None:
                images = visualize_matching_helper(np.copy(im_list[0]), point2d_cloud[ind],
                                                   point3d_cloud[point_ind], sfm_images_folder)
                cv2.imshow("t", images)
                cv2.waitKey()
                cv2.destroyWindow("t")
            fx, fy = point2d_cloud[ind].xy
            fx, fy = map(int, (fx//down_scale, fy//down_scale))
            cv2.circle(image, (fx, fy), 5, (128, 128, 0), 5)
        elif MODE == 2:
            if mouseX < 0:
                continue
            if (mouseX, mouseY) not in RESULT_NO_RATIO:
                _, ind = point2d_cloud.xy_tree.query([mouseX, mouseY], 1)
                data1 = point3d_cloud.matching_2d_to_3d_brute_force_no_ratio_test(point2d_cloud[ind].desc)
                point_ind, dist, _, ratio = data1
                RESULT_NO_RATIO[(mouseX, mouseY)] = (ind, point_ind, dist)
                print(f"ratio test: {mouseX, mouseY} => {ind, point_ind, dist, ratio}")
            else:
                ind, point_ind, dist = RESULT_NO_RATIO[(mouseX, mouseY)]
            if point_ind is not None:
                images = visualize_matching_helper(np.copy(im_list[0]), point2d_cloud[ind],
                                                   point3d_cloud[point_ind], sfm_images_folder)
                cv2.imshow("t", images)
                cv2.waitKey()
                cv2.destroyWindow("t")
            fx, fy = point2d_cloud[ind].xy
            fx, fy = map(int, (fx // down_scale, fy // down_scale))
            cv2.circle(image, (fx, fy), 5, (128, 128, 0), 5)
            MODE = None
            image = np.copy(image_ori)
            mouseX = -1
            mouseY = -1

        k = cv2.waitKey(20) & 0xFF

        if k == 27 or k == ord("q"):
            break
        elif k == ord('a'):
            MODE = 1
        elif k == ord('b'):
            MODE = 2
        elif k == ord('d'):
            image = np.copy(image_ori)
            mouseX = -1
            mouseY = -1
            MODE = None
            RESULT = {}

    cv2.destroyAllWindows()
else:
    _, ind = point2d_cloud.xy_tree.query([mouseX, mouseY], 1)
    fx, fy = point2d_cloud[ind].xy

    fx, fy = map(int, (fx // down_scale, fy // down_scale))
    cv2.circle(image, (fx, fy), 5, (128, 128, 0), 5)
    cv2.circle(image, (mouseX//down_scale, mouseY//down_scale), 5, (255, 0, 0), 5)

    cv2.imshow("image", image)
    point_ind, dist, _ = point3d_cloud.matching_2d_to_3d_brute_force(point2d_cloud[ind].desc,
                                                                     returning_index=True)
    if point_ind is not None:
        visualize_matching([(point2d_cloud[ind], None, dist)],
                           [(point2d_cloud[ind], point3d_cloud[point_ind], dist)],
                           im_list[0], sfm_images_folder)