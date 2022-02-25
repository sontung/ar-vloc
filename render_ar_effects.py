import numpy as np
import trimesh
import pyrender

import colmap_io
import colmap_read
import open3d as o3d
import open3d.visualization.rendering as rendering
from vis_utils import produce_cam_mesh, produce_proj_mat_4


def produce_mat(data):
    qw, qx, qy, qz, tx, ty, tz = data
    ref_rot_mat = colmap_read.qvec2rotmat([qw, qx, qy, qz])
    t_vec = np.array([tx, ty, tz])
    t_vec = t_vec.reshape((-1, 1))
    rot_mat, trans = (ref_rot_mat, t_vec)
    rot_mat = -rot_mat.transpose()
    t = rot_mat @ trans
    t = t.reshape((3, 1))
    mat_ = np.hstack([rot_mat, t])
    mat_ = np.vstack([mat_, np.array([0, 0, 0, 1])])
    return mat_


def produce_o3d_cam(mat):
    camera_parameters = o3d.camera.PinholeCameraParameters()
    width = 1920
    height = 1025
    focal = 0.9616278814278851
    k_mat = [[focal * width, 0, width / 2 - 0.5],
             [0, focal * width, height / 2 - 0.5],
             [0, 0, 1]]
    camera_parameters.extrinsic = mat
    camera_parameters.intrinsic.set_intrinsics(width=width, height=height,
                                               fx=k_mat[0][0], fy=k_mat[1][1],
                                               cx=k_mat[0][2], cy=k_mat[1][2])

    return camera_parameters


def main(sfm_images_dir, sfm_point_cloud_dir):
    image2pose_gt = colmap_io.read_images(sfm_images_dir)
    name2pose_gt = {}
    for im_id in image2pose_gt:
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose_gt[im_id]
        name2pose_gt[image_name] = cam_pose
    point3did2xyzrgb = colmap_io.read_points3D_coordinates(sfm_point_cloud_dir)
    points_3d_list = []
    for point3d_id in point3did2xyzrgb:
        x, y, z, r, g, b = point3did2xyzrgb[point3d_id]
        points_3d_list.append([x, y, z, r/255, g/255, b/255])
    points_3d_list = np.vstack(points_3d_list)
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d_list[:, :3]))
    point_cloud.colors = o3d.utility.Vector3dVector(points_3d_list[:, 3:])
    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1025)

    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pose_list = [[-0.227121, 0.959549, -0.0284974, 0.163917, -0.234621, -2.2819, 1.03221],
                 [-0.196202, 0.967265, -0.0278312, 0.158523, -0.368659, -2.14311, 2.10415],
                 [0.0404636, 0.988771, -0.0153438, 0.143038, -0.398162, -0.372246, 3.57212]]

    mesh = o3d.io.read_triangle_mesh("data/teapot.obj")
    mesh.scale(0.1, mesh.get_center())
    mesh.translate([0, 0, 0], relative=False)
    mat = produce_mat(pose_list[0])
    vertices = np.asarray(mesh.vertices)
    for i in range(vertices.shape[0]):
        arr = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1])
        arr = mat @ arr
        vertices[i] = arr[:3]
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.translate([0, -1.2, -2.1], relative=True)
    mesh.compute_triangle_normals()
    vis.add_geometry(point_cloud, reset_bounding_box=True)
    vis.add_geometry(mesh, reset_bounding_box=True)
    cam_correct = produce_cam_mesh((1, 0, 0), mat=mat)
    vis.add_geometry(cam_correct)

    mat = produce_proj_mat_4(pose_list[0])
    camera_parameters = produce_o3d_cam(mat)
    cam_vis = o3d.geometry.LineSet.create_camera_visualization(camera_parameters.intrinsic, camera_parameters.extrinsic)
    vis.add_geometry(cam_vis)

    # vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameters)

    # while True:
    #     vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameters)
    #     param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    #     # print(param.extrinsic)
    #     vis.poll_events()
    #     vis.update_renderer()
    vis.run()
    vis.destroy_window()


# main2()
# ref_test()
main("/home/sontung/work/recon_models/indoor/sparse/images.txt",
     "/home/sontung/work/recon_models/indoor/sparse/points3D.txt")