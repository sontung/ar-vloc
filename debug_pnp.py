import colmap_read
import retrieval_based_pycolmap
import numpy as np
import cv2
from visloc_pseudo_gt_limitations import evaluation_util as vis_loc_pseudo_eval_utils


def opencv_pnp(pairs, metadata):
    camera_matrix = np.array([
        [metadata["f"], 0, metadata["cx"]],
        [0, metadata["f"], metadata["cy"]],
        [0, 0, 1]
    ])
    image_points = []
    object_points = []
    for xy, xyz, _ in pairs:
        image_points.append(xy)
        object_points.append(xyz)
    object_points = np.array(object_points).reshape((-1, 1, 3))
    image_points = np.array(image_points).reshape((-1, 1, 2))
    print(image_points.dtype)

    val, rot, trans, inliers = cv2.solvePnPRansac(object_points, image_points,
                                                  camera_matrix, distCoeffs=None,
                                                  flags=cv2.SOLVEPNP_SQPNP)
    mask = np.zeros((image_points.shape[0],))
    mask[inliers[:, 0]] = 1
    if not val:
        return None
    rot_mat, _ = cv2.Rodrigues(rot)
    return rot_mat, trans


def main():
    f = open("7scenes_ws_div/logs.txt", "r")
    lines = f.readlines()
    pairs = []
    for line in lines:
        line = line[:-1]
        xy0, xy1, xyz0, xyz1, xyz2, dis = map(float, line.split(" "))
        pairs.append(((xy0, xy1), (xyz0, xyz1, xyz2), dis))
    best_score = None
    best_pose = None
    best_diff = None
    metadata = {'f': 525.505, 'cx': 320.0, 'cy': 240.0, "h": 640, "w": 480}

    gt_file = "visloc_pseudo_gt_limitations/pgt/sfm/7scenes/redkitchen_test.txt"
    pgt_poses = vis_loc_pseudo_eval_utils.read_pose_data(gt_file)
    query_im_name = "seq-14/frame-000258.color.png"
    pgt_pose, rgb_focal_length = pgt_poses[query_im_name]
    data = []

    for _ in range(10):
        r_mat, t_vec, score, mask, diff = retrieval_based_pycolmap.localize(metadata, [(du1, du2) for du1, du2, _ in pairs])
        qw, qx, qy, qz = colmap_read.rotmat2qvec(r_mat)
        tx, ty, tz = t_vec[:, 0]
        result = f"{query_im_name} {qw} {qx} {qy} {qz} {tx} {ty} {tz}"

        est_poses = vis_loc_pseudo_eval_utils.convert_pose_data([result])
        est_pose, est_f = est_poses[query_im_name]
        error = vis_loc_pseudo_eval_utils.compute_error_max_rot_trans(pgt_pose, est_pose)
        data.append([error, mask])
        print(mask.shape)
        total = 0
        for idx in np.nonzero(mask)[0]:
            total += pairs[idx][-1]

        r_mat, t_vec = opencv_pnp(pairs, metadata)
        qw, qx, qy, qz = colmap_read.rotmat2qvec(r_mat)
        tx, ty, tz = t_vec[:, 0]
        result = f"{query_im_name} {qw} {qx} {qy} {qz} {tx} {ty} {tz}"

        est_poses = vis_loc_pseudo_eval_utils.convert_pose_data([result])
        est_pose, est_f = est_poses[query_im_name]
        error2 = vis_loc_pseudo_eval_utils.compute_error_max_rot_trans(pgt_pose, est_pose)

        print(error, error2, np.sum(mask), total)

    bad_mask = max(data, key=lambda du: du[0])[1]
    good_mask = min(data, key=lambda du: du[0])[1]
    print(max([du[0] for du in data]), min([du[0] for du in data]))
    in_good_but_not_in_bad = []
    in_bad_but_not_in_good = []

    for i, m in enumerate(good_mask):
        if m:
            in_good_but_not_in_bad.append(i)
        elif bad_mask[i]:
            in_bad_but_not_in_good.append(i)
    print(in_good_but_not_in_bad)
    print(np.nonzero(good_mask))
    print(np.nonzero(bad_mask))

    arr = []
    img = cv2.imread("/media/sontung/580ECE740ECE4B28/7_scenes_images/redkitchen/seq-14/frame-000258.color.png")
    total = 0
    for idx in np.nonzero(good_mask)[0]:
        xy = pairs[idx][0]
        arr.append(xy)
        x2, y2 = map(int, xy)
        cv2.circle(img, (x2, y2), 5, (255, 0, 0), -1)
        total += pairs[idx][-1]

    # cv2.imshow("", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    cv2.imwrite("7scenes_ws_div/good.png", img)
    print(np.var(arr, axis=0), total)

    arr = []
    total = 0
    img = cv2.imread("/media/sontung/580ECE740ECE4B28/7_scenes_images/redkitchen/seq-14/frame-000258.color.png")
    for idx in np.nonzero(bad_mask)[0]:
        xy = pairs[idx][0]
        arr.append(xy)
        x2, y2 = map(int, xy)
        cv2.circle(img, (x2, y2), 5, (0, 255, 0), -1)
        total += pairs[idx][-1]

    # cv2.imshow("", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    cv2.imwrite("7scenes_ws_div/bad.png", img)

    print(np.var(arr, axis=0), total)

    # r_mat, t_vec = best_pose
    # qw, qx, qy, qz = colmap_read.rotmat2qvec(r_mat)
    # tx, ty, tz = t_vec[:, 0]
    # result = f"{query_im_name} {qw} {qx} {qy} {qz} {tx} {ty} {tz}"
    #
    # est_poses = vis_loc_pseudo_eval_utils.convert_pose_data([result])
    # est_pose, est_f = est_poses[query_im_name]
    # error = vis_loc_pseudo_eval_utils.compute_error_max_rot_trans(pgt_pose, est_pose)
    # print("final", error)

if __name__ == '__main__':
    main()