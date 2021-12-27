import cv2
import numpy as np
import time
import pnp.build.pnp_python_binding
import kmeans1d
import sklearn.cluster


def localize_shared_intrinsics(p2d2p3d, coord_list, point3did2xyzrgb, camera_matrix, distortion_coefficients):
    """
    using pnp algorithm to compute camera pose
    """
    results = []
    for im_idx in p2d2p3d:
        pairs = p2d2p3d[im_idx]
        object_points = []
        image_points = []
        for point2d_id, point3d_id in pairs:
            coord_2d = coord_list[im_idx][point2d_id]
            coord_3d = point3did2xyzrgb[point3d_id][:3]
            image_points.append(coord_2d)
            object_points.append(coord_3d)
        object_points = np.array(object_points)
        image_points = np.array(image_points).reshape((-1, 1, 2))

        val, rot, trans, inliers = cv2.solvePnPRansac(object_points, image_points,
                                                      camera_matrix, distortion_coefficients)
        if not val:
            print(f"{object_points.shape[0]} 2D-3D pairs computed but localization failed.")
            results.append(None)
            continue
        rot_mat, _ = cv2.Rodrigues(rot)
        results.append([rot_mat, trans])
        print(f"{inliers.shape[0]}/{image_points.shape[0]} are inliers")
    return results


def localize_single_image(pairs, camera_matrix, distortion_coefficients):
    """
    using pnp algorithm to compute camera pose
    """
    start_time = time.time()
    object_points = []
    image_points = []
    for xy, xyz in pairs:
        image_points.append(xy)
        object_points.append(xyz)
    object_points = np.array(object_points)
    image_points = np.array(image_points).reshape((-1, 1, 2))

    val, rot, trans, inliers = cv2.solvePnPRansac(object_points, image_points,
                                                  camera_matrix, distortion_coefficients,
                                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not val:
        print(f" {object_points.shape[0]} 2D-3D pairs computed but localization failed.")
        return None
    rot_mat, _ = cv2.Rodrigues(rot)
    print(f" {inliers.shape[0]}/{image_points.shape[0]} are inliers, "
          f"time spent={round(time.time()-start_time, 4)} seconds")
    return rot_mat, trans


def localize_single_image_lt_pnp(pairs):
    object_points = []
    image_points = []
    for xy, xyz in pairs:
        image_points.append(xy)
        object_points.append(xyz)
    object_points = np.array(object_points)
    image_points = np.array(image_points)
    res = pnp.build.pnp_python_binding.pnp(object_points, image_points)
    return res


if __name__ == '__main__':
    xs = np.array([[-17.8431, 0.570044, 11.1874], [-80.6362, -23.8517, 21.0087], [-68.0126, 9.19776, 20.6913],
                   [-8.31825, -13.5394, 23.8776], [-32.3177, 30.9775, 35.0005], [-60.5264, 3.64722, 62.0491],
                   [-13.8288, -0.638686, 30.1851], [-25.1182, 35.7954, 81.3263], [0.841874, -20.8397, 42.3626],
                   [-2.04336, 0.61477, 0.620302]])
    ys = np.array([[-0.083742, 0.314872], [-0.516025, 0.0535602], [-0.392733, 0.51515], [0.400942, -0.423236],
                   [0.371449, 0.98387], [0.123111, 0.257844], [0.481032, 0.102744], [0.850471, 0.608635],
                   [0.846186, -0.652791], [0.154041, 0.784826]])
    localize_single_image_lt_pnp(xs, ys)