import random
import sys

import cv2
import numpy as np
import tqdm

import colmap_io
import colmap_read
import open3d as o3d
from vis_utils import produce_cam_mesh, produce_proj_mat_4, produce_mat, make_video


def produce_o3d_cam(mat, width, height):
    camera_parameters = o3d.camera.PinholeCameraParameters()
    # width = 1920
    # height = 1025
    focal = 0.9616278814278851
    k_mat = [[focal * width, 0, width / 2 - 0.5],
             [0, focal * width, height / 2 - 0.5],
             [0, 0, 1]]
    camera_parameters.extrinsic = mat
    camera_parameters.intrinsic.set_intrinsics(width=width, height=height,
                                               fx=k_mat[0][0], fy=k_mat[1][1],
                                               cx=k_mat[0][2], cy=k_mat[1][2])

    return camera_parameters


def produce_object(color=None):
    mesh = o3d.io.read_triangle_mesh("data/teapot.obj")
    if color is None:
        color = [random.random(), random.random(), random.random()]
    mesh.paint_uniform_color(color)
    rot_mat = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 2))
    mesh.rotate(rot_mat, mesh.get_center())
    # mesh.scale(random.uniform(0.01, 0.05), mesh.get_center())
    mesh.scale(0.03, mesh.get_center())

    if color != [0, 0, 0]:
        mesh.compute_vertex_normals()
    return mesh


def sample_scene(vis, point_cloud, first_pass):
    color = None
    if first_pass:
        color = [0, 0, 0]
    mesh = produce_object(color)
    mesh.translate(point_cloud.get_center(), relative=False)
    vis.add_geometry(mesh, reset_bounding_box=True)
    mesh2 = produce_object(color)
    mesh2.translate(point_cloud.get_center(), relative=False)
    mesh2.translate([0, 1, 0.5])
    vis.add_geometry(mesh2, reset_bounding_box=True)
    mesh3 = produce_object(color)
    mesh3.translate(point_cloud.get_center(), relative=False)
    mesh3.translate([0, -1, -0.5])
    vis.add_geometry(mesh3, reset_bounding_box=True)

    mesh4 = produce_object(color)
    mesh4.translate(point_cloud.get_center(), relative=False)
    mesh4.translate([0, 0, -2.5])
    vis.add_geometry(mesh4, reset_bounding_box=True)
    mesh5 = produce_object(color)
    mesh5.translate(point_cloud.get_center(), relative=False)
    mesh5.translate([0, 1, -2.5])
    vis.add_geometry(mesh5, reset_bounding_box=True)
    mesh6 = produce_object(color)
    mesh6.translate(point_cloud.get_center(), relative=False)
    mesh6.translate([0, -1, -2.5])
    vis.add_geometry(mesh6, reset_bounding_box=True)


def render(sfm_images_dir, sfm_point_cloud_dir, vid, no_point_cloud=True, first_pass=True):
    width = 1920
    height = 1080
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
    vis.create_window(width=width, height=height, visible=False)

    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    sample_scene(vis, point_cloud, first_pass)

    if not no_point_cloud:
        vis.add_geometry(point_cloud, reset_bounding_box=True)
        coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        vis.add_geometry(coord_mesh, reset_bounding_box=True)

    number_to_image_id = {}
    for image_id in image2pose_gt:
        image_name = image2pose_gt[image_id][0]
        number = int(image_name.split("-")[-1].split(".")[0])
        number_to_image_id[number] = image_id
    image_seq = sorted(list(number_to_image_id.keys()))

    for number2 in range(10, 900, 30):
        number = image_seq[number2]
        image_id = number_to_image_id[number]
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose_gt[image_id]

        mat = produce_proj_mat_4(cam_pose)
        camera_parameters = produce_o3d_cam(mat, width, height)
        cam_vis = o3d.geometry.LineSet.create_camera_visualization(camera_parameters.intrinsic,
                                                                   camera_parameters.extrinsic)
        # vis.add_geometry(cam_vis)
        if first_pass:
            mesh_new = produce_object([0, 0, 0])
        else:
            mesh_new = produce_object()

        mesh_new.translate(cam_vis.get_center(), relative=False)
        vis.add_geometry(mesh_new)

    for number2, number in enumerate(tqdm.tqdm(image_seq)):
        image_id = number_to_image_id[number]
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose_gt[image_id]
        mat = produce_proj_mat_4(cam_pose)
        camera_parameters = produce_o3d_cam(mat, width, height)

        # int_mat = camera_parameters.intrinsic.intrinsic_matrix
        # print(int_mat[0, 2], width/2-0.5, int_mat[1, 2], height/2-0.5, camera_parameters.intrinsic.width,
        #       camera_parameters.intrinsic.height)
        # camera_parameters2 = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # print(camera_parameters2.intrinsic.width, camera_parameters2.intrinsic.height)
        # sys.exit()

        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameters)

        vis.poll_events()
        vis.update_renderer()

        if first_pass:
            vis.capture_screen_image(f"{vid}/img-{number2}-2.png", True)
        else:
            vis.capture_screen_image(f"{vid}/img-{number2}.png", True)

    # make_video(vid, fps=15)
    # vis.run()
    vis.destroy_window()


def project_to_video(sfm_images_dir, sfm_point_cloud_dir, sfm_images_folder, rendering_folder, augmented_folder):
    # render(sfm_images_dir, sfm_point_cloud_dir, rendering_folder)
    # render(sfm_images_dir, sfm_point_cloud_dir, rendering_folder, first_pass=False)

    image2pose_gt = colmap_io.read_images(sfm_images_dir)
    name2pose_gt = {}
    for im_id in image2pose_gt:
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose_gt[im_id]
        name2pose_gt[image_name] = cam_pose

    number_to_image_id = {}
    for image_id in image2pose_gt:
        image_name = image2pose_gt[image_id][0]
        number = int(image_name.split("-")[-1].split(".")[0])
        number_to_image_id[number] = image_id
    image_seq = sorted(list(number_to_image_id.keys()))

    for number2 in tqdm.tqdm(range(len(image_seq))):
        number = image_seq[number2]
        image_id = number_to_image_id[number]
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose_gt[image_id]
        img = cv2.imread(f"{sfm_images_folder}/{image_name}")
        mask = cv2.imread(f"{rendering_folder}/img-{number2}-2.png")
        augment = cv2.imread(f"{rendering_folder}/img-{number2}.png")
        mask2 = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
        mask3 = np.ones((mask.shape[0], mask.shape[1]), np.uint8)
        mask3[mask[:, :, 0] == 0] = 0
        mask2[mask[:, :, 0] == 0] = 1
        img = np.multiply(img, mask3[:, :, None])
        augment2 = np.multiply(augment, mask2[:, :, None])
        final = img + augment2
        cv2.imwrite(f"{augmented_folder}/img-{number2}.png", final)


if __name__ == '__main__':
    project_to_video("/home/sontung/work/sparse_outdoor/images.txt",
                     "/home/sontung/work/sparse_outdoor/points3D.txt",
                     "/home/sontung/work/sparse_outdoor/images/images",
                     "/home/sontung/work/ar-vloc/data/augment_video",
                     "/home/sontung/work/ar-vloc/data/ar_video")
    make_video("/home/sontung/work/ar-vloc/data/ar_video", 15)
