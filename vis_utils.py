import open3d as o3d
import random
import sys
import numpy as np
import time
import cv2
import os
import glob
import json
import tqdm
import os
import moviepy.video.io.ImageSequenceClip
import colmap_read
from colmap_io import read_images, read_points3D_coordinates
from PIL import Image
from scipy.spatial import KDTree
from pathlib import Path


def produce_cam_mesh(color=None, res=4, mat=None):
    camera_mesh2 = o3d.geometry.TriangleMesh.create_cone(resolution=res)
    camera_mesh2.scale(0.25, camera_mesh2.get_center())
    # camera_mesh2.translate([0, 0, 0], relative=False)

    if color is not None:
        camera_mesh2.paint_uniform_color(color)

    if mat is not None:
        vertices = np.asarray(camera_mesh2.vertices)
        for i in range(vertices.shape[0]):
            arr = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1])
            arr = mat @ arr
            vertices[i] = arr[:3]
        camera_mesh2.vertices = o3d.utility.Vector3dVector(vertices)
    return camera_mesh2


def produce_sphere(pos, color=None):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    mesh.compute_vertex_normals()
    mesh.translate(pos, relative=False)

    if color:
        mesh.paint_uniform_color(color)
    return mesh


def matching_2d_to_3d_single(point3d_desc_list, desc):
    kd_tree = KDTree(point3d_desc_list)
    distances, indices = kd_tree.query(desc, 10)
    return distances, indices


def visualize_2d_3d_matching_single(p2d2p3d, coord_2d_list, im_name_list,
                                    point3did2xyzrgb, original_point_cloud,
                                    query_folder, point3d_id_list, point3d_desc_list, desc_list):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1025, visible=False)
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2021-11-24-10-37-09.json")

    for image_idx in p2d2p3d:
        vis.add_geometry(original_point_cloud)
        point_cloud = []
        good_matches = 0
        all_matches = []
        print(f"visualizing {len(p2d2p3d[image_idx])} pairs")
        # while True:
        for p2d_id, p3d_id in p2d2p3d[image_idx]:
            # p2d_id, p3d_id = random.choice(p2d2p3d[image_idx])
            # if p2d_id not in [5395]:
            #     continue

            distances, indices = matching_2d_to_3d_single(point3d_desc_list, desc_list[image_idx][p2d_id])
            all_matches.append([distances[0], p2d_id])
            print(f"point 2d id={p2d_id}, distances={distances}")
            # if distances[0] < 200:
            #     continue
            good_matches += 1
            meshes = []
            img = cv2.imread(f"{query_folder}/{im_name_list[image_idx]}")

            p2d_coord, p3d_coord = coord_2d_list[image_idx][p2d_id], point3did2xyzrgb[p3d_id][:3]
            v, u = map(int, p2d_coord)
            point_cloud.append(p3d_coord)
            color = [0, 0, 0]
            mesh = produce_sphere(p3d_coord, color)
            meshes.append(mesh)
            cv2.circle(img, (v, u), 20, [0, 0, 0], -1)

            for index in indices[1:]:
                p3d_id = point3d_id_list[index]
                color = [random.random() for _ in range(3)]
                mesh = produce_sphere(point3did2xyzrgb[p3d_id][:3], color)
                meshes.append(mesh)

            for m in meshes:
                vis.add_geometry(m, reset_bounding_box=False)

            ctr.convert_from_pinhole_camera_parameters(parameters)
            ctr.set_zoom(0.5)
            vis.capture_screen_image("trash_code/test.png", do_render=True)

            img_vis = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
            for m in meshes:
                vis.remove_geometry(m, reset_bounding_box=False)
            cv2.imshow("t", img_vis)
            cv2.waitKey()
            cv2.destroyAllWindows()

        print(good_matches)
        print(sorted(all_matches, key=lambda du: du[0]))
        break
    vis.destroy_window()


def visualize_matching_helper(query_image, feature, point, sfm_image_folder):
    visualized_list = [query_image]
    for database_image in point.visibility:
        x2, y2 = map(int, point.visibility[database_image])
        image = cv2.imread(f"{sfm_image_folder}/{database_image}")
        cv2.circle(image, (x2, y2), 50, (128, 128, 0), -1)
        visualized_list.append(image)
    list2 = []
    for im in visualized_list:
        im2 = Image.fromarray(im)
        im2.thumbnail((500, 500))
        im2 = np.array(im2)
        list2.append(im2)
    list2 = np.hstack(list2)
    return list2


def visualize_all_point_images(point, sfm_image_folder):
    visualized_list = []
    for database_image in point.visibility:
        x2, y2 = map(int, point.visibility[database_image])
        image = cv2.imread(f"{sfm_image_folder}/{database_image}")
        cv2.circle(image, (x2, y2), 50, (128, 128, 0), -1)
        visualized_list.insert(0, image)
    list2 = []
    for im in visualized_list:
        im2 = Image.fromarray(im)
        im2.thumbnail((500, 500))
        im2 = np.array(im2)
        list2.append(im2)
    list2 = np.hstack(list2)
    return list2


def visualize_matching(bf_results, results, query_image_ori, sfm_image_folder):
    for ind in range(len(results)):
        # print(results[ind][0].xy, bf_results[ind][0].xy)
        assert np.sum(results[ind][0].xy-bf_results[ind][0].xy) < 0.1
        feature, point, dist1 = results[ind]
        query_image = np.copy(query_image_ori)
        l1 = visualize_matching_helper(query_image, feature, point, sfm_image_folder)
        cv2.imshow("t", l1)

        feature, point, dist2 = bf_results[ind]
        if point is not None:
            query_image = np.copy(query_image_ori)
            l2 = visualize_matching_helper(query_image, feature, point, sfm_image_folder)
            cv2.imshow("t2", l2)
        # print(f"vc dist1={dist1} bf dist={dist2}")
        cv2.waitKey()
        cv2.destroyWindow("t2")


def visualize_matching_and_save(results, query_image_ori, sfm_image_folder, debug_dir, folder_name):
    os.makedirs(f"{debug_dir}/{folder_name}", exist_ok=True)
    files = glob.glob(f"{debug_dir}/{folder_name}/*.png")
    for f in files:
        os.remove(f)
    for ind in range(len(results)):
        feature, point, dist1 = results[ind]
        query_image = np.copy(query_image_ori)
        l1 = visualize_matching_helper(query_image, feature, point, sfm_image_folder)
        cv2.imwrite(f"{debug_dir}/{folder_name}/im-{ind}.png", l1)


def visualize_matching_pairs(image1, image2, pairs):
    image = np.hstack([image1, image2])
    for pair in pairs:
        fid1, fid2 = pair[:2]
        x1, y1 = fid1
        cv2.circle(image, (x1, y1), 20, (128, 128, 0), -1)

        x2, y2 = fid2
        cv2.circle(image, (x2 + image1.shape[1], y2), 20, (128, 128, 0), -1)
        cv2.line(image, (x1, y1), (x2 + image1.shape[1], y2), (0, 0, 0), 5)
    return image


def visualize_cam_pose_with_point_cloud(point_cloud, localization_results):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1025)

    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    cameras = [point_cloud, coord_mesh]

    # queried poses
    for result, color_cam in localization_results:
        if result is None:
            continue
        rot_mat, trans = result
        t = -rot_mat.transpose() @ trans
        t = t.reshape((3, 1))
        mat = np.hstack([-rot_mat.transpose(), t])
        mat = np.vstack([mat, np.array([0, 0, 0, 1])])

        cm = produce_cam_mesh(color=color_cam)

        vertices = np.asarray(cm.vertices)
        for i in range(vertices.shape[0]):
            arr = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1])
            arr = mat @ arr
            vertices[i] = arr[:3]
        cm.vertices = o3d.utility.Vector3dVector(vertices)
        cameras.append(cm)

    for c in cameras:
        vis.add_geometry(c)
    vis.run()
    vis.destroy_window()


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


def produce_proj_mat(data):
    qw, qx, qy, qz, tx, ty, tz = data
    ref_rot_mat = colmap_read.qvec2rotmat([qw, qx, qy, qz])
    t_vec = np.array([tx, ty, tz])
    t_vec = t_vec.reshape((3, 1))

    mat_ = np.hstack([ref_rot_mat, t_vec])
    return mat_


def produce_proj_mat_4(data):
    qw, qx, qy, qz, tx, ty, tz = data
    ref_rot_mat = colmap_read.qvec2rotmat([qw, qx, qy, qz])
    t_vec = np.array([tx, ty, tz])
    t_vec = t_vec.reshape((3, 1))

    mat_ = np.hstack([ref_rot_mat, t_vec])
    mat_ = np.vstack([mat_, np.array([0, 0, 0, 1])])
    return mat_


def produce_o3d_cam(mat=None, cam_intrinsic=None):
    camera_parameters = o3d.camera.PinholeCameraParameters()
    if cam_intrinsic is None:
        width = 1920
        height = 1025
        focal = 0.9616278814278851
    else:
        width, height, focal = cam_intrinsic

    if mat is None:
        f = open('view.json')
        data = json.load(f)
        mat = np.array(data["extrinsic"]).reshape((4, 4)).T
        K = np.array(data["intrinsic"]["intrinsic_matrix"]).reshape((3, 3)).T

        camera_parameters.extrinsic = mat
        camera_parameters.intrinsic.set_intrinsics(
            width=width, height=height, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2])
        pass
    else:
        K = [[focal * width, 0, width / 2-0.5],
             [0, focal * width, height / 2-0.5],
             [0, 0, 1]]

        camera_parameters.extrinsic = mat
    camera_parameters.intrinsic.set_intrinsics(
        width=width, height=height, fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2])
    return camera_parameters


def visualize_reconstruction_process(sfm_image_dir, sfm_point_cloud_dir,
                                     vid="/home/sontung/work/ar-vloc/data/indoor_video"):
    point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)

    image2pose = read_images(sfm_image_dir)
    number_to_image_id = {}
    for image_id in image2pose:
        image_name = image2pose[image_id][0]
        number = 0
        try:
            number = int(image_name.split("-")[-1].split(".")[0])
        except ValueError:
            if "image" == image_name[:5]:
                number = int(image_name[5:].split(".")[0])
        number_to_image_id[number] = image_id

    image_seq = sorted(list(number_to_image_id.keys()))

    points_3d_list = []
    point_cloud = None
    point_id_to_point_index = {}
    for number in image_seq:
        image_id = number_to_image_id[number]
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose[image_id]

        for _, _, point3d_id in points2d_meaningful:
            if point3d_id != -1 and point3d_id not in point_id_to_point_index:
                x, y, z, r, g, b = point3did2xyzrgb[point3d_id]
                point_id_to_point_index[point3d_id] = [len(points_3d_list), r / 255, g / 255, b / 255]
                points_3d_list.append([x, y, z, 1, 1, 1])
    points_3d_arr = np.vstack(points_3d_list)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1025)
    vis.get_render_option().point_size = 2
    vis.get_render_option().background_color = np.array([1, 1, 1])
    camera_parameters = produce_o3d_cam(None)
    vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameters)

    for number2, number in enumerate(tqdm.tqdm(image_seq)):
        if point_cloud is not None:
            vis.remove_geometry(point_cloud, reset_bounding_box=True)
        image_id = number_to_image_id[number]
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose[image_id]

        for _, _, point3d_id in points2d_meaningful:
            if point3d_id != -1:
                idx, r, g, b = point_id_to_point_index[point3d_id]
                points_3d_arr[idx, 3:] = [r, g, b]
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d_arr[:, :3]))
        point_cloud.colors = o3d.utility.Vector3dVector(points_3d_arr[:, 3:])

        vis.add_geometry(point_cloud, reset_bounding_box=True)

        if number2 % 4 == 0:
            mat = produce_mat(cam_pose)
            cm = produce_cam_mesh(color=(1, 0, 0))
            vertices = np.asarray(cm.vertices)
            for i in range(vertices.shape[0]):
                arr = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1])
                arr = mat @ arr
                vertices[i] = arr[:3]
            cm.vertices = o3d.utility.Vector3dVector(vertices)
            cm2 = o3d.geometry.LineSet.create_from_triangle_mesh(cm)
            vis.add_geometry(cm, reset_bounding_box=True)
            vis.add_geometry(cm2, reset_bounding_box=True)

        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameters)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(f"{vid}/img-{number2}.png")
    # make_video(vid)
    vis.run()
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("view.json", param)
    vis.destroy_window()
    return


def make_video(image_folder, fps=50):
    image_files = [os.path.join(image_folder, img)
                   for img in os.listdir(image_folder)
                   if img.endswith(".png")]
    image_files = sorted(image_files, key=lambda du: int(du.split("/")[-1].split("-")[-1].split(".")[0]))
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(f'{image_folder}/my_video.mp4')


def visualize_camera_sequence(sfm_image_dir, sfm_point_cloud_dir):
    point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)

    image2pose = read_images(sfm_image_dir)
    number_to_image_id = {}
    for image_id in image2pose:
        image_name = image2pose[image_id][0]
        number = 0
        try:
            number = int(image_name.split("-")[-1].split(".")[0])
        except ValueError:
            if "image" == image_name[:5]:
                number = int(image_name[5:].split(".")[0])
        number_to_image_id[number] = image_id
    image_seq = sorted(list(number_to_image_id.keys()))

    points_3d_list = []
    point_id_to_point_index = {}
    for number in image_seq:
        image_id = number_to_image_id[number]
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose[image_id]

        for _, _, point3d_id in points2d_meaningful:
            if point3d_id != -1 and point3d_id not in point_id_to_point_index:
                x, y, z, r, g, b = point3did2xyzrgb[point3d_id]
                point_id_to_point_index[point3d_id] = [len(points_3d_list), r / 255, g / 255, b / 255]
                points_3d_list.append([x, y, z, r / 255, g / 255, b / 255])
    points_3d_arr = np.vstack(points_3d_list)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1025)
    vis.get_render_option().point_size = 2.5
    vis.get_render_option().background_color = np.array([1, 1, 1])
    camera_parameters2 = produce_o3d_cam(None)
    vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameters2)

    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d_arr[:, :3]))
    point_cloud.colors = o3d.utility.Vector3dVector(points_3d_arr[:, 3:])
    vis.add_geometry(point_cloud)

    for number2, number in enumerate(tqdm.tqdm(image_seq)):
        image_id = number_to_image_id[number]
        image_name, points2d_meaningful, cam_pose, cam_id = image2pose[image_id]

        mat = produce_mat(cam_pose)
        cm = produce_cam_mesh(color=(1, 0, 0))
        vertices = np.asarray(cm.vertices)
        for i in range(vertices.shape[0]):
            arr = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1])
            arr = mat @ arr
            vertices[i] = arr[:3]
        cm.vertices = o3d.utility.Vector3dVector(vertices)
        cm = o3d.geometry.LineSet.create_from_triangle_mesh(cm)
        vis.add_geometry(cm, reset_bounding_box=True)

        mat = produce_proj_mat_4(cam_pose)
        camera_parameters = produce_o3d_cam(mat)
        cam_vis = o3d.geometry.LineSet.create_camera_visualization(camera_parameters.intrinsic,
                                                                   camera_parameters.extrinsic)
        vis.add_geometry(cam_vis)
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameters2)

        vis.poll_events()
        vis.update_renderer()
    vis.run()
    # param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()
    return


if __name__ == '__main__':
    # produce_o3d_cam(None)
    make_video("/home/sontung/work/ar-vloc/data/indoor_video")
    # visualize_reconstruction_process("/home/sontung/work/recon_models/indoor/images.txt",
    #                                  "/home/sontung/work/recon_models/indoor/points3D.txt",
    #                                  "/home/sontung/work/ar-vloc/data/indoor_video")
    # visualize_camera_sequence("/home/sontung/work/recon_models/indoor/images.txt",
    #                           "/home/sontung/work/recon_models/indoor/points3D.txt")
