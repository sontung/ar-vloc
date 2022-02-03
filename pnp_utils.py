import pnp.build.pnp_python_binding
import numpy as np


def compute_smoothness_cost_pnp(solution, pid_coord_list, fid_coord_list, f, c1, c2, threshold=0.001):
    if f == 0 and c1 == 0 and c2 == 0:
        raise TypeError
    object_points = []
    image_points = []
    object_points_homo = []
    object_indices = []
    image_indices = []

    for u, v in enumerate(solution):
        if v is not None:
            object_indices.append(u)
            image_indices.append(v)
            xyz = pid_coord_list[u]
            xy = fid_coord_list[v]
            x, y = xy
            u2 = (x - c1) / f
            v2 = (y - c2) / f
            image_points.append([u2, v2])
            object_points.append(xyz)
            x, y, z = xyz
            object_points_homo.append([x, y, z, 1.0])
    object_points = np.array(object_points)
    object_points_homo = np.array(object_points_homo)
    image_points = np.array(image_points)
    object_indices = np.array(object_indices)
    image_indices = np.array(image_indices)

    if object_points.shape[0] < 4:
        return -1, [], []
    mat = pnp.build.pnp_python_binding.pnp(object_points, image_points)

    xy = mat[None, :, :] @ object_points_homo[:, :, None]
    xy = xy[:, :, 0]
    xy = xy[:, :3] / xy[:, 2].reshape((-1, 1))
    xy = xy[:, :2]
    diff = np.sum(np.square(xy - image_points), axis=1)
    inliers = np.sum(diff < threshold)
    return inliers, object_indices, image_indices


def compute_smoothness_cost_pnp2(solution, point3d_cloud, point2d_cloud, f, c1, c2, threshold=0.001):
    if f == 0 and c1 == 0 and c2 == 0:
        raise TypeError
    object_points = []
    image_points = []
    object_points_homo = []
    object_indices = []
    image_indices = []

    for u, v in solution:
        object_indices.append(u)
        image_indices.append(v)
        xyz = point3d_cloud[u].xyz
        xy = point2d_cloud[v].xy
        x, y = xy
        u2 = (x - c1) / f
        v2 = (y - c2) / f
        image_points.append([u2, v2])
        object_points.append(xyz)
        x, y, z = xyz
        object_points_homo.append([x, y, z, 1.0])
    object_points = np.array(object_points)
    object_points_homo = np.array(object_points_homo)
    image_points = np.array(image_points)
    object_indices = np.array(object_indices)
    image_indices = np.array(image_indices)

    if object_points.shape[0] < 4:
        return -1, [], []
    mat = pnp.build.pnp_python_binding.pnp(object_points, image_points)

    xy = mat[None, :, :] @ object_points_homo[:, :, None]
    xy = xy[:, :, 0]
    xy = xy[:, :3] / xy[:, 2].reshape((-1, 1))
    xy = xy[:, :2]
    diff = np.sum(np.square(xy - image_points), axis=1)
    inliers = np.sum(diff < threshold)
    object_indices, image_indices = object_indices[diff < threshold], image_indices[diff < threshold]
    return inliers, object_indices, image_indices


def filter_bad_matches(pid_coord_list, fid_coord_list, f, c1, c2, threshold=0.001):
    if f == 0 and c1 == 0 and c2 == 0:
        raise TypeError
    object_points = []
    image_points = []
    object_points_homo = []

    for ind in range(len(pid_coord_list)):
        xyz = pid_coord_list[ind]
        xy = fid_coord_list[ind]
        x, y = xy
        u2 = (x - c1) / f
        v2 = (y - c2) / f
        image_points.append([u2, v2])
        object_points.append(xyz)
        x, y, z = xyz
        object_points_homo.append([x, y, z, 1.0])
    object_points = np.array(object_points)
    object_points_homo = np.array(object_points_homo)
    image_points = np.array(image_points)

    if object_points.shape[0] < 4:
        return -1, [], []
    mat = pnp.build.pnp_python_binding.pnp(object_points, image_points)

    xy = mat[None, :, :] @ object_points_homo[:, :, None]
    xy = xy[:, :, 0]
    xy = xy[:, :3] / xy[:, 2].reshape((-1, 1))
    xy = xy[:, :2]
    diff = np.sum(np.square(xy - image_points), axis=1)
    cond = diff < threshold
    indices = [ind for ind in range(len(pid_coord_list)) if cond[ind]]
    return indices
