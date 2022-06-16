import pathlib
import pickle
import time
import poselib
import cv2
import numpy as np
import pnp.build.pnp_python_binding
from tqdm import tqdm


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


def localize_single_image_opencv(pairs, f, c1, c2):
    """
    using pnp algorithm to compute camera pose
    """
    object_points = []
    image_points = []
    camera_matrix = np.array([
        [f, 0, c1],
        [0, f, c2],
        [0, 0, 1]
    ])
    for xy, xyz in pairs:
        image_points.append(xy)
        object_points.append(xyz)
    object_points = np.array(object_points).reshape((-1, 1, 3))
    image_points = np.array(image_points).reshape((-1, 1, 2)).astype(np.float64)
    # print(image_points.dtype)
    mask = np.zeros((image_points.shape[0],))

    val, rot, trans, inliers = cv2.solvePnPRansac(object_points, image_points,
                                                  camera_matrix, distCoeffs=None,
                                                  iterationsCount=500,
                                                  reprojectionError=0.1,
                                                  flags=cv2.SOLVEPNP_SQPNP)
    if not val:
        return np.identity(3), np.array([0, 0, 0]).reshape((-1, 1)), 0, mask
    rot_mat, _ = cv2.Rodrigues(rot)
    mask[inliers[:, 0]] = 1
    tqdm.write(
        f" localization is done with {inliers.shape[0]}/{image_points.shape[0]}"
        f" inliers ({round(inliers.shape[0] / image_points.shape[0], 3)})")
    return rot_mat, trans, inliers.shape[0] / image_points.shape[0], mask


def localize_single_image_opencv_refine(pairs, f, c1, c2, init_pose, ransac=True):
    """
    using pnp algorithm to compute camera pose
    """
    object_points = []
    image_points = []
    camera_matrix = np.array([
        [f, 0, c1],
        [0, f, c2],
        [0, 0, 1]
    ])
    for xy, xyz in pairs:
        image_points.append(xy)
        object_points.append(xyz)
    object_points = np.array(object_points).reshape((-1, 1, 3))
    image_points = np.array(image_points).reshape((-1, 1, 2)).astype(np.float64)

    init_rot_mat, init_trans = init_pose
    init_rot_mat = cv2.Rodrigues(init_rot_mat)[0]

    if ransac:
        mask = np.zeros((image_points.shape[0],))
        print(init_rot_mat, init_trans)
        val, rot, trans, inliers = cv2.solvePnPRansac(object_points, image_points,
                                                      camera_matrix, distCoeffs=None,
                                                      useExtrinsicGuess=True,
                                                      rvec=init_rot_mat, tvec=init_trans,
                                                      reprojectionError=5,
                                                      flags=cv2.SOLVEPNP_ITERATIVE)
        print(val)
        print(init_rot_mat, init_trans)
        print(rot, trans)
        if not val:
            return np.identity(3), np.array([0, 0, 0]).reshape((-1, 1)), 0, mask
        rot_mat, _ = cv2.Rodrigues(rot)
        mask[inliers[:, 0]] = 1
        tqdm.write(
            f" localization is done with {inliers.shape[0]}/{image_points.shape[0]}"
            f" inliers ({round(inliers.shape[0] / image_points.shape[0], 3)})")
        return rot_mat, trans, inliers.shape[0] / image_points.shape[0], mask
    else:
        rot, trans = cv2.solvePnPRefineLM(object_points, image_points,
                                          camera_matrix, distCoeffs=None,
                                          rvec=init_rot_mat, tvec=init_trans)
        rot_mat, _ = cv2.Rodrigues(rot)
        return rot_mat, trans


def localize_single_image_lt_pnp(pairs, f, c1, c2, with_inliers_percent=False, return_inlier_mask=True):
    object_points = []
    image_points = []
    object_points_homo = []

    for xy, xyz in pairs:
        x, y = xy
        u = (x - c1) / f
        v = (y - c2) / f
        image_points.append([u, v])
        object_points.append(xyz)
        x, y, z = xyz
        object_points_homo.append([x, y, z, 1.0])
    if len(object_points) <= 3:
        if with_inliers_percent:
            return np.identity(3), np.array([0, 0, 0]).reshape((-1, 1)), 0, [True] * len(object_points)
        return np.identity(3), np.array([0, 0, 0]).reshape((-1, 1))
    object_points = np.array(object_points)
    image_points = np.array(image_points)
    threshold = 1 / f
    res = pnp.build.pnp_python_binding.pnp(object_points, image_points, threshold)

    object_points_homo = np.array(object_points_homo)
    xy = res[None, :, :] @ object_points_homo[:, :, None]
    xy = xy[:, :, 0]
    xy = xy[:, :3] / xy[:, 2].reshape((-1, 1))
    xy = xy[:, :2]
    diff = np.sum(np.square(xy - image_points), axis=1)
    inliers = np.sum(diff < threshold ** 2)
    # tqdm.write(
    #     f" localization is done with diff={round(float(np.sum(diff)), 3)} {inliers}/{image_points.shape[0]}"
    #     f" inliers ({round(inliers / image_points.shape[0], 3)})")

    # return in opencv format
    r_mat, t_vec = res[:3, :3], res[:3, 3]
    t_vec = t_vec.reshape((-1, 1))
    if return_inlier_mask:
        return r_mat, t_vec, inliers / image_points.shape[0], diff < threshold ** 2, diff
    if with_inliers_percent:
        return r_mat, t_vec, inliers / image_points.shape[0]

    return r_mat, t_vec


def localize_pose_lib(pairs, f, c1, c2):
    """
    using pose lib to compute (usually best)
    """
    camera = {'model': 'SIMPLE_PINHOLE', 'height': int(c1*2), 'width': int(c2*2), 'params': [f, c1, c2]}
    object_points = []
    image_points = []
    for xy, xyz in pairs:
        xyz = np.array(xyz).reshape((3, 1))
        xy = np.array(xy)
        xy = xy.reshape((2, 1)).astype(np.float64)
        image_points.append(xy)
        object_points.append(xyz)
    pose, info = poselib.estimate_absolute_pose(image_points, object_points, camera, {'max_reproj_error': 16.0}, {})
    return pose, info
