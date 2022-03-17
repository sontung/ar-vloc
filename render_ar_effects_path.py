import random
import sys
import cv2
import numpy as np
import tqdm
import colmap_io
import open3d as o3d

import math_utils
from vis_utils import produce_o3d_cam, produce_proj_mat_4, make_video, read_image_sequence, produce_mat, \
    produce_cam_mesh


def produce_object(color=None):
    mesh = o3d.geometry.TriangleMesh.create_sphere()
    if color is None:
        color = [random.random(), random.random(), random.random()]
    mesh.paint_uniform_color(color)
    rot_mat = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rot_mat, mesh.get_center())
    # mesh.scale(random.uniform(0.01, 0.05), mesh.get_center())
    mesh.scale(0.007, mesh.get_center())
    mesh.translate([0, 0, 0], relative=False)
    if np.sum(color) > 0:
        mesh.compute_vertex_normals()
    return mesh


def produce_object2(color=None):
    mesh = o3d.geometry.TriangleMesh.create_icosahedron()
    if color is None:
        color = [random.random(), random.random(), random.random()]
    mesh.paint_uniform_color(color)
    rot_mat = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rot_mat, mesh.get_center())
    # mesh.scale(random.uniform(0.01, 0.05), mesh.get_center())
    mesh.scale(0.01, mesh.get_center())
    mesh.translate([0, 0, 0], relative=False)
    if np.sum(color) > 0:
        mesh.compute_vertex_normals()
    return mesh


def sample_scene(vis, point_cloud, first_pass):
    color = None
    if first_pass:
        color = [0, 0, 0]
    mesh = produce_object(color)
    mesh.translate(point_cloud.get_center(), relative=False)
    mesh.translate([0, 0, -0.5])

    vis.add_geometry(mesh, reset_bounding_box=True)


def render(point_cloud, image2pose_gt, vid, first_pass=True, debug=False):
    width = 1920
    height = 1080

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    # vis.create_window()

    image_seq, number_to_image_id = read_image_sequence(image2pose_gt)

    # add arrows
    image_seq = image_seq[:583]
    last_pose = None
    for number2, number in enumerate(tqdm.tqdm(image_seq)):
        image_id = number_to_image_id[number]
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose_gt[image_id]
        # last_pose = cam_pose
        if number2 % 12 == 0:
            mat = produce_mat(cam_pose)
            if 500 <= number2 <= 525 or number2+12 >= len(image_seq):
                continue
            else:
                if first_pass:
                    cm = produce_object(color=[0, 0, 0])
                else:
                    if number2 == 0:
                        cm = produce_object(color=[1, 0, 0])
                    else:
                        cm = produce_object(color=[0, 0, 1])

            cm.transform(mat)
            # print(number2)
            # if number2 > 400:
            vis.add_geometry(cm, reset_bounding_box=True)
            last_pose = cam_pose

    mat = produce_mat(last_pose)
    if first_pass:
        cm = read_mesh(pose=mat, color=[0, 0, 0])
    else:
        cm = read_mesh(pose=mat, color=[1, 0, 0])
    cm.translate(last_pose[-3:], relative=False)

    vis.add_geometry(cm, reset_bounding_box=True)

    # mat = produce_proj_mat_4(last_pose)
    # camera_parameters = produce_o3d_cam(mat, width, height)
    # vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameters)

    # cm2 = read_mesh(color=[1, 0, 0])
    # vertices = np.asarray(cm2.vertices)
    # for i in range(vertices.shape[0]):
    #     arr = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1])
    #     arr = mat @ arr
    #     vertices[i] = arr[:3]
    # cm2.vertices = o3d.utility.Vector3dVector(vertices)
    # vis.add_geometry(cm2, reset_bounding_box=True)

    # vis.run()
    # vis.destroy_window()
    # sys.exit()

    # render
    for number2, number in enumerate(tqdm.tqdm(image_seq)):
        image_id = number_to_image_id[number]
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose_gt[image_id]
        mat = produce_proj_mat_4(cam_pose)
        camera_parameters = produce_o3d_cam(mat, width, height)

        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameters)

        vis.poll_events()
        vis.update_renderer()

        if first_pass:
            vis.capture_screen_image(f"{vid}/img-{number2}-2.png", True)
        else:
            vis.capture_screen_image(f"{vid}/img-{number2}.png", True)

    vis.destroy_window()


def project_to_video(sfm_images_dir, sfm_point_cloud_dir, sfm_images_folder, rendering_folder, augmented_folder):
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

    render(point_cloud, image2pose_gt, rendering_folder, debug=True, first_pass=False)

    render(point_cloud, image2pose_gt, rendering_folder)
    render(point_cloud, image2pose_gt, rendering_folder, first_pass=False)

    image_seq, number_to_image_id = read_image_sequence(image2pose_gt)
    image_seq = image_seq[:583]

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


def read_mesh2():
    mesh = o3d.io.read_triangle_mesh("data/navigation_target.ply")
    mesh.paint_uniform_color([1, 0, 0])
    # mesh.compute_vertex_normals()

    vertices = np.copy(np.asarray(mesh.vertices))
    normals = np.copy(np.asarray(mesh.vertex_normals))
    point_before = o3d.geometry.PointCloud(mesh.vertices)
    point_before.paint_uniform_color([0, 0, 1])
    point_before2 = o3d.geometry.PointCloud(mesh.vertex_normals)
    point_before2.paint_uniform_color([0, 1, 1])

    mat = np.array([[1.0, - 0.0, - 0.0, 0.0],
                    [0.0, -1, - 0.0, 100.0],
                    [-0.0, 0.0, 1, 0.0],
                    [0., 0., 0., 1.]])

    # mat = np.array([[-0.96721006, - 0.15799728, - 0.19885059, 50.27780248],
    #                 [0.13407839, 0.34731152, - 0.92811513, 30.81414389],
    #                 [-0.21570277, 0.92434386, 0.31473918, 10.12006442],
    #                 [0., 0., 0., 1.]])

    # for i in range(vertices.shape[0]):
    #     arr = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1])
    #     arr = mat @ arr
    #     vertices[i] = arr[:3]
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # mat2 = np.transpose(np.linalg.inv(mat))
    # tri_normals = np.asarray(mesh.triangle_normals)
    # for i in range(tri_normals.shape[0]):
    #     a2 = math_utils.multDirMatrix(tri_normals[i], mat2[:3, :3])
    #     arr = np.array([tri_normals[i, 0],
    #                     tri_normals[i, 1],
    #                     tri_normals[i, 2], 0])
    #     arr = arr @ mat2
    #     tri_normals[i] = -a2
    # mesh.triangle_normals = o3d.utility.Vector3dVector(tri_normals)
    mesh.transform(mat)
    # mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    points = []
    lines = []
    for i in range(vertices.shape[0]):
        points.append(vertices[i])
        points.append(normals[i]*2+vertices[i])
        lines.append([i * 2, i * 2 + 1])
    line_set = o3d.geometry.LineSet(o3d.utility.Vector3dVector(points), o3d.utility.Vector2iVector(lines))
    point1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
    point1.paint_uniform_color([1, 0, 0])
    point2 = o3d.geometry.PointCloud(mesh.vertex_normals)
    point2.paint_uniform_color([0, 1, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point1)
    vis.add_geometry(point_before)
    vis.add_geometry(point_before2)

    vis.add_geometry(point2)
    vis.add_geometry(line_set)

    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()
    sys.exit()
    return mesh


def read_mesh(pose=None, color=None):
    mesh = o3d.io.read_triangle_mesh("data/navigation_target.ply")
    mesh.paint_uniform_color(color)
    mesh.scale(0.01/2, mesh.get_center())
    # rot_mat = mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    # mesh.rotate(rot_mat, mesh.get_center())
    # if pose is not None:
    #     mesh.transform(pose)
    rot_mat = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi/2))
    mesh.rotate(rot_mat, mesh.get_center())
    return mesh


if __name__ == '__main__':
    # read_mesh2()
    # project_to_video("/home/sontung/work/recon_models/indoor_all/sparse/images.txt",
    #                  "/home/sontung/work/recon_models/indoor_all/sparse/points3D.txt",
    #                  "/home/sontung/work/recon_models/indoor_all/images",
    #                  "/home/sontung/work/ar-vloc/data/augment_video",
    #                  "/home/sontung/work/ar-vloc/data/ar_video")
    make_video("/home/sontung/work/ar-vloc/data/ar_video", 15)
