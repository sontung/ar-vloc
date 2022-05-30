import colmap_read
import retrieval_based_pycolmap
from visloc_pseudo_gt_limitations import evaluation_util as vis_loc_pseudo_eval_utils

f = open("7scenes_ws_div/logs.txt", "r")
lines = f.readlines()
pairs = []
for line in lines:
    line = line[:-1]
    xy0, xy1, xyz0, xyz1, xyz2 = map(float, line.split(" "))
    pairs.append(((xy0, xy1), (xyz0, xyz1, xyz2)))
best_score = None
best_pose = None
best_diff = None
metadata = {'f': 525.505 / 100, 'cx': 320.0, 'cy': 240.0, "h": 640, "w": 480}

gt_file = "visloc_pseudo_gt_limitations/pgt/sfm/7scenes/redkitchen_test.txt"
pgt_poses = vis_loc_pseudo_eval_utils.read_pose_data(gt_file)
query_im_name = "seq-14/frame-000258.color.png"
pgt_pose, rgb_focal_length = pgt_poses[query_im_name]

for _ in range(10):
    r_mat, t_vec, score, mask, diff = retrieval_based_pycolmap.localize(metadata, pairs)

    qw, qx, qy, qz = colmap_read.rotmat2qvec(r_mat)
    tx, ty, tz = t_vec[:, 0]
    result = f"{query_im_name} {qw} {qx} {qy} {qz} {tx} {ty} {tz}"

    est_poses = vis_loc_pseudo_eval_utils.convert_pose_data([result])
    est_pose, est_f = est_poses[query_im_name]
    error = vis_loc_pseudo_eval_utils.compute_error_max_rot_trans(pgt_pose, est_pose)
    print(error)

    if best_score is None or score > best_score:
        best_score = score
        best_pose = (r_mat, t_vec)
        best_diff = diff

r_mat, t_vec = best_pose
qw, qx, qy, qz = colmap_read.rotmat2qvec(r_mat)
tx, ty, tz = t_vec[:, 0]
result = f"{query_im_name} {qw} {qx} {qy} {qz} {tx} {ty} {tz}"

est_poses = vis_loc_pseudo_eval_utils.convert_pose_data([result])
est_pose, est_f = est_poses[query_im_name]
error = vis_loc_pseudo_eval_utils.compute_error_max_rot_trans(pgt_pose, est_pose)
print("final", error)
