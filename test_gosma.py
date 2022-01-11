import numpy as np
import imageio
import cv2
import random
from colmap_io import read_points3D_coordinates
from feature_matching import compute_kp_descriptors_opencv_with_d2_detector, load_2d_queries_generic

query_images_folder = "Test line small"
descriptors, coordinates, im_names, md_list, im_list, response_list = load_2d_queries_generic(query_images_folder)
coords, _ = compute_kp_descriptors_opencv_with_d2_detector(im_list[0], nb_keypoints=8000)
im_show = np.copy(im_list[0])
for x1, y1 in coords:
    color = (random.random() * 255, random.random() * 255, random.random() * 255)
    x1, y1 = map(int, (x1, y1))
    cv2.circle(im_show, (x1, y1), 20, color, -1)
# im_show = cv2.resize(im_show, (im_show.shape[1] // 4, im_show.shape[0] // 4))
# cv2.imshow("t", im_show)
# cv2.waitKey()
# cv2.destroyAllWindows()

new_coords = []
f, c1, c2 = 2600.0, 1134.0, 2016.0
for x1, y1 in coords:
    u = (x1 - c1) / f
    v = (y1 - c2) / f
    new_coords.append([u, v, 1])
new_coords = np.array(new_coords)
np.savetxt("debug/image.txt", new_coords)

object_coords = read_points3D_coordinates("sfm_ws_hblab/points3D.txt")
object_coords_arr = [object_coords[du][:3] for du in object_coords]
object_coords_arr = np.array(object_coords_arr)
np.savetxt("debug/object.txt", object_coords_arr)

# ext_mat = [[0.254249, -0.346227,  0.903042, -1.256218],
#            [0.189263, -0.897860, -0.397527, -1.531903],
#            [0.948439,  0.271983, -0.162752, 4.787511],
#            [0, 0, 0, 1]
#            ]
#
# res_line = "-1.256218195e+00 -1.531902909e+00  4.787511349e+00  2.542491555e-01  1.892632991e-01  9.484391212e-01 -3.462268710e-01 -8.978596330e-01  2.719834745e-01  9.030417204e-01 -3.975266516e-01 -1.627520323e-01 8.271529526e-02 3.937936290e-01 8.734280000e-04 1"
# numbers = []
# for du in res_line.split(" "):
#     try:
#         n = float(du)
#     except ValueError:
#         continue
#     else:
#         numbers.append(n)
# trans_mat = np.identity(4).astype(np.float64)
# trans_mat[0, 3] = -numbers[0]
# trans_mat[1, 3] = -numbers[1]
# trans_mat[2, 3] = -numbers[2]
# rot_mat = np.identity(4).astype(np.float64)
# rot_mat[:3, 0] = numbers[3:6]
# rot_mat[:3, 1] = numbers[6:9]
# rot_mat[:3, 2] = numbers[9:12]
# res_mat = rot_mat@trans_mat
#
# print(res_mat)
#
# image_mat = np.loadtxt("debug/image.txt")
# object_mat = np.loadtxt("debug/object.txt")
#
# for idx, xyz in enumerate(object_mat):
#     xyz2 = rot_mat[:3, :3]@(xyz-numbers[:3])
#
#     xyz = np.hstack([xyz, 1])
#     xyz = res_mat@xyz
#
#     u, v, w = xyz[:3]
#     u /= w
#     v /= w
#     uv = np.array([u, v, 1])
#     test = image_mat-uv
#     test = test[:, :2]
#     test = np.square(test)
#     test = np.sum(test, axis=1)
#     if np.min(test) < 0.1:
#         print(idx)
