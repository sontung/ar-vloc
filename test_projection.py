import numpy as np

import colmap_io
from vis_utils import produce_mat, produce_proj_mat, produce_proj_mat_4


def radial_world_to_image(u, v, f, c1, c2, k):
    u2 = u*u
    v2 = v*v
    r2 = u2+v2
    radial = k*r2
    du = u*radial
    dv = v*radial
    x = u+du
    y = v+dv
    x = f*x+c1
    y = f*y+c2
    return x, y


def main():
    sfm_images_dir = "/home/sontung/work/recon_models/indoor/sparse/images.txt"
    sfm_point_cloud_dir = "/home/sontung/work/recon_models/indoor/sparse/points3D.txt"
    image2pose_gt = colmap_io.read_images(sfm_images_dir)
    name2pose_gt = {}
    point3did2xyzrgb = colmap_io.read_points3D_coordinates(sfm_point_cloud_dir)

    f, c1, c2, k = 1672.36, 960, 540, 0.0403787
    image_points = []
    object_points = []
    object_points_homo = []

    for im_id in image2pose_gt:
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose_gt[im_id]
        proj_mat = produce_proj_mat(cam_pose)
        ext_mat = produce_proj_mat_4(cam_pose)
        for x, y, pid in points2d_meaningful:
            if pid != -1:
                px, py, pz = point3did2xyzrgb[pid][:3]
                image_points.append([x, y])
                object_points.append([px, py, pz])
                object_points_homo.append([px, py, pz, 1])
                homo = np.array([px, py, pz, 1])
                projected_homo = proj_mat @ homo
                projected_homo2 = ext_mat @ homo
                print(projected_homo2, projected_homo)

                projected_homo /= projected_homo[2]
        break


if __name__ == '__main__':
    main()