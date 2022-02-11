import open3d as o3d
import random
import sys
import numpy as np
import time
import cv2
import os
import glob

from PIL import Image
from scipy.spatial import KDTree


def produce_cam_mesh(color=None, res=4):
    camera_mesh2 = o3d.geometry.TriangleMesh.create_cone(resolution=res)
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
    visualized_list = []
    (x, y) = map(int, feature.xy)
    # cv2.circle(query_image, (x, y), 50, (128, 128, 0), -1)

    visualized_list.append(query_image)
    for database_image in point.visibility:
        x2, y2 = map(int, point.visibility[database_image])
        image = cv2.imread(f"{sfm_image_folder}/{database_image}")
        cv2.circle(image, (x2, y2), 50, (128, 128, 0), 20)
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