import sys

import numpy as np
import pickle
import os
import torch
import cv2
import kornia
import random
import psutil

import open3d as o3d
from feature_matching import compute_kp_descriptors, build_descriptors_2d, load_2d_queries_opencv
from colmap_io import build_descriptors, read_points3D_coordinates, read_images
from colmap_read import qvec2rotmat
from pathlib import Path
from scipy.spatial import KDTree
from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as rot_mat_compute

NEXT = False
DEBUG_2D_3D_MATCHING = False
DEBUG_PNP = True


def load_2d_queries(folder="test_images"):
    im_names_dir = "test_images/im_names.pkl"
    desc_dir = 'test_images/keypoint_descriptors.pt'
    coord_dir = 'test_images/keypoint_coordinates.pt'
    my_file = Path(im_names_dir)
    if my_file.is_file():
        print("Loading 2D descriptors for test images at test_images/")
        descs = torch.load(desc_dir)
        coordinates = torch.load(coord_dir)
        with open("test_images/im_names.pkl", "rb") as fp:
            im_names = pickle.load(fp)
    else:
        im_names = os.listdir(folder)
        descs = []
        coordinates = []
        for name in im_names:
            im_name = os.path.join(folder, name)

            im = cv2.imread(im_name)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2))
            coord, desc = compute_kp_descriptors(im)

            # im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)/255.0
            # im = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2))
            # im_list.append(np.expand_dims(im, -1).astype(float))
            # an_im = kornia.utils.image_list_to_tensor(im_list)
            # with torch.no_grad():
            #     laf, response, desc = sift_model.forward(an_im)
            #     descs.append(desc.cpu())
            #     points = kornia.feature.laf.get_laf_center(laf)

            descs.append(desc)
            coordinates.append(coord)
        descs = torch.stack(descs)
        coordinates = torch.stack(coordinates)
        torch.save(coordinates, coord_dir)
        torch.save(descs, desc_dir)
        with open(im_names_dir, "wb") as fp:
            pickle.dump(im_names, fp)
        print("Saved 2D descriptors at test_images/")

    return descs, coordinates, im_names


def load_3d_database():
    point3d_ids_dir = "data/point3d_ids.pkl"
    point3d_descs_dir = "data/point3d_descs"

    my_file = Path(point3d_ids_dir)
    if my_file.is_file():
        print("Loading 3D descriptors at data/")
        point3d_desc_list = np.load(f"{point3d_descs_dir}.npy")
        with open(point3d_ids_dir, "rb") as fp:
            point3d_id_list = pickle.load(fp)
        print(f"\t{len(point3d_id_list)} 3D points with desc mat {point3d_desc_list.shape}")
    else:
        os.makedirs("data", exist_ok=True)
        _, point3did2descs = build_descriptors()
        point3d_id_list = []
        point3d_desc_list = []
        for point3d_id in point3did2descs:
            point3d_id_list.append(point3d_id)
            descs = [data[1] for data in point3did2descs[point3d_id]]
            mean_desc = torch.mean(torch.stack(descs), 0)
            point3d_desc_list.append(mean_desc.numpy())
        point3d_desc_list = np.vstack(point3d_desc_list)
        print("Saved 3D descriptors at data/")
        np.save(point3d_descs_dir, point3d_desc_list)
        with open(point3d_ids_dir, "wb") as fp:
            pickle.dump(point3d_id_list, fp)
    return point3d_id_list, point3d_desc_list


def matching_2d_to_3d(point3d_id_list, point3d_desc_list, point2d_desc_list):
    """
    returns [image id] => point 2d id => point 3d id
    """
    kd_tree = KDTree(point3d_desc_list)
    result = {i: [] for i in range(len(point2d_desc_list))}
    for i in range(len(point2d_desc_list)):
        desc_list = point2d_desc_list[i]
        for j in range(desc_list.shape[0]):
            desc = desc_list[j]
            res = kd_tree.query(desc, 2)
            if res[0][1] > 0.0:
                if res[0][0]/res[0][1] < 0.7:  # ratio test
                    result[i].append([j, point3d_id_list[res[1][0]]])
    return result


def matching_2d_to_3d_single(point3d_desc_list, desc):
    kd_tree = KDTree(point3d_desc_list)
    distances, indices = kd_tree.query(desc, 10)
    return distances, indices


def key_sw(u):
    global NEXT
    NEXT = not NEXT


def visualize_2d_3d_matching(p2d2p3d, coord_2d_list, im_name_list, point3did2xyzrgb, original_point_cloud):
    global NEXT

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1920, height=1025)
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2021-11-24-10-37-09.json")
    vis.register_key_callback(ord("N"), key_sw)

    for image_idx in p2d2p3d:
        NEXT = False
        vis.add_geometry(original_point_cloud)

        img = cv2.imread(f"test_images/{im_name_list[image_idx]}")

        point_cloud = []
        colors = []
        meshes = []
        print(f"visualizing {len(p2d2p3d[image_idx])} pairs")
        for p2d_id, p3d_id in p2d2p3d[image_idx]:
            p2d_coord, p3d_coord = coord_2d_list[image_idx][p2d_id], point3did2xyzrgb[p3d_id][:3]
            v, u = map(int, p2d_coord)
            point_cloud.append(p3d_coord)
            colors.append([random.random() for _ in range(3)])
            mesh = produce_sphere(p3d_coord, colors[-1])
            cv2.circle(img, (v, u), 5, np.array(colors[-1])*255, -1)

            meshes.append(mesh)

        for m in meshes:
            vis.add_geometry(m)
        ctr.convert_from_pinhole_camera_parameters(parameters)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        vis.capture_screen_image("trash_code/test.png", do_render=True)
        Image.fromarray(img).show()
        while True:
            vis.poll_events()
            vis.update_renderer()
            if NEXT:
                vis.clear_geometries()
                for proc in psutil.process_iter():
                    if proc.name() == "display":
                        proc.kill()
                break
    vis.destroy_window()


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


def produce_cam_mesh(color=None, res=4):
    camera_mesh2 = o3d.geometry.TriangleMesh.create_cone(resolution=res)
    camera_mesh2.compute_vertex_normals()
    camera_mesh2.scale(0.25, camera_mesh2.get_center())
    camera_mesh2.translate([0, 0, 0], relative=False)

    if color:
        camera_mesh2.paint_uniform_color(color)
    return camera_mesh2


def produce_sphere(pos, color=None):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    mesh.compute_vertex_normals()
    mesh.translate(pos, relative=False)

    if color:
        mesh.paint_uniform_color(color)
    return mesh


def main():
    query_images_folder = "test_images"
    image2pose = read_images()
    point3did2xyzrgb = read_points3D_coordinates()
    points_3d_list = []
    point3d_id_list, point3d_desc_list = build_descriptors_2d()

    for point3d_id in point3did2xyzrgb:
        x, y, z, r, g, b = point3did2xyzrgb[point3d_id]
        if point3d_id in point3d_id_list:
            point3did2xyzrgb[point3d_id] = [x, y, z, 255, 0, 0]
        else:
            point3did2xyzrgb[point3d_id] = [x, y, z, 0, 0, 0]

    for point3d_id in point3did2xyzrgb:
        x, y, z, r, g, b = point3did2xyzrgb[point3d_id]
        points_3d_list.append([x, y, z, r/255, g/255, b/255])
    points_3d_list = np.vstack(points_3d_list)
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d_list[:, :3]))
    point_cloud.colors = o3d.utility.Vector3dVector(points_3d_list[:, 3:])
    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

    # desc_list, coord_list, im_name_list = load_2d_queries()
    # point3d_id_list, point3d_desc_list = load_3d_database()

    desc_list, coord_list, im_name_list = load_2d_queries_opencv(query_images_folder)
    p2d2p3d = matching_2d_to_3d(point3d_id_list, point3d_desc_list, desc_list)

    if DEBUG_2D_3D_MATCHING:
        # visualize_2d_3d_matching(p2d2p3d, coord_list, im_name_list, point3did2xyzrgb, point_cloud)
        visualize_2d_3d_matching_single(p2d2p3d, coord_list, im_name_list,
                                        point3did2xyzrgb, point_cloud, query_images_folder,
                                        point3d_id_list, point3d_desc_list, desc_list)

    if DEBUG_PNP:
        f, cx, cy, k = 1596.1472395458961, 575.0, 1024.0, 2.8106522618619115e-05
        camera_matrix = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
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
            print(object_points.shape, image_points.shape)

            cv2.solvePnPRansac(object_points, image_points, camera_matrix, np.array([k, k]))
        sys.exit()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1025)
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2021-11-22-16-30-08.json")

    cm1 = produce_cam_mesh([0.5, 1, 0.5], 20)
    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    cameras = [point_cloud, coord_mesh, cm1]
    for image_id in image2pose:
        pose1 = np.array(image2pose[image_id][2])
        trans1 = np.array(pose1[4:])
        rot_mat1 = qvec2rotmat(pose1[:4])

        t = -rot_mat1.transpose()@trans1
        t = t.reshape((3, 1))
        mat = np.hstack([-rot_mat1.transpose(), t])
        mat = np.vstack([mat, np.array([0, 0, 0, 1])])

        cm = produce_cam_mesh()

        vertices = np.asarray(cm.vertices)
        for i in range(vertices.shape[0]):
            arr = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1])
            arr = mat@arr
            vertices[i] = arr[:3]
        cm.vertices = o3d.utility.Vector3dVector(vertices)
        cameras.append(cm)
        # break

    for c in cameras:
        vis.add_geometry(c)
    ctr.convert_from_pinhole_camera_parameters(parameters)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
