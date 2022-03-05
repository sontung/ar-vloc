import random
import sys
import trimesh
import cv2
import numpy as np
import tqdm

import colmap_io
import colmap_read
import open3d as o3d
from vis_utils import produce_cam_mesh, produce_proj_mat_4, produce_mat, make_video
import trimesh.exchange


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
    rot_mat = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi))
    mesh.rotate(rot_mat, mesh.get_center())
    # mesh.scale(random.uniform(0.01, 0.05), mesh.get_center())
    mesh.scale(0.3, mesh.get_center())
    if color != [0, 0, 0]:
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
    width = 1080
    height = 1920

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=debug)

    sample_scene(vis, point_cloud, first_pass)

    number_to_image_id = {}
    for image_id in image2pose_gt:
        image_name = image2pose_gt[image_id][0]
        number = int(image_name.split("-")[-1].split(".")[0])
        number_to_image_id[number] = image_id
    image_seq = sorted(list(number_to_image_id.keys()))

    if debug:
        vis.add_geometry(point_cloud, reset_bounding_box=True)
        coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        vis.add_geometry(coord_mesh, reset_bounding_box=True)
        number = image_seq[-1]
        image_id = number_to_image_id[number]
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose_gt[image_id]
        mat = produce_proj_mat_4(cam_pose)
        camera_parameters = produce_o3d_cam(mat, width, height)

        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameters)

        vis.run()
        vis.destroy_window()
        sys.exit()

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

    # render(point_cloud, image2pose_gt, rendering_folder, debug=True)

    render(point_cloud, image2pose_gt, rendering_folder)
    render(point_cloud, image2pose_gt, rendering_folder, first_pass=False)

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
    # project_to_video("/home/sontung/work/recon_models/bookshelf/images.txt",
    #                  "/home/sontung/work/recon_models/bookshelf/points3D.txt",
    #                  "/home/sontung/work/recon_models/bookshelf/images",
    #                  "/home/sontung/work/ar-vloc/data/augment_video",
    #                  "/home/sontung/work/ar-vloc/data/ar_video")
    make_video("/home/sontung/work/ar-vloc/data/ar_video", 15)
