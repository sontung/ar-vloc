import cv2
import numpy as np
import time
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


def localize_single_image(pairs, coord_list, point3did2xyzrgb, camera_matrix, distortion_coefficients):
    """
    using pnp algorithm to compute camera pose
    """
    start_time = time.time()
    object_points = []
    image_points = []
    for point2d_id, point3d_id in pairs:
        coord_2d = coord_list[point2d_id]
        coord_3d = point3did2xyzrgb[point3d_id][:3]
        image_points.append(coord_2d)
        object_points.append(coord_3d)
    object_points = np.array(object_points)
    image_points = np.array(image_points).reshape((-1, 1, 2))

    val, rot, trans, inliers = cv2.solvePnPRansac(object_points, image_points,
                                                  camera_matrix, distortion_coefficients)
    if not val:
        print(f"{object_points.shape[0]} 2D-3D pairs computed but localization failed.")
        return None
    rot_mat, _ = cv2.Rodrigues(rot)
    print(f"{inliers.shape[0]}/{image_points.shape[0]} are inliers, "
          f"time spent={round(time.time()-start_time, 4)} seconds")
    return rot_mat, trans

