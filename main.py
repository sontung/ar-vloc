import numpy as np
import pickle
import os
import torch
import cv2
import kornia
import open3d as o3d
from colmap_io import build_descriptors, read_points3D_coordinates, read_images
from colmap_read import qvec2rotmat
from pathlib import Path
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as rot_mat_compute



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
        sift_model = kornia.feature.SIFTFeature(num_features=1000, device=torch.device("cpu"))
        descs = []
        coordinates = []
        for name in im_names:
            im_list = []
            im_name = os.path.join(folder, name)
            im = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, (im.shape[1]//2, im.shape[0]//2))
            im_list.append(np.expand_dims(im, -1).astype(float))
            # cv2.imshow("t", im)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            an_im = kornia.utils.image_list_to_tensor(im_list)
            with torch.no_grad():
                laf, response, desc = sift_model.forward(an_im)
                descs.append(desc.cpu())
                points = kornia.feature.laf.get_laf_center(laf)
                coordinates.append(points)
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


def matching_2d_to_3d(point3d_id_list, point3d_desc_list):
    kd_tree = KDTree(point3d_desc_list)
    for i in range(point3d_desc_list.shape[0]):
        desc = point3d_desc_list[i]
        res = kd_tree.query(desc, 2)
        # print(res[0][0], res[1][0])
        # break


def produce_cam_mesh(color=None, res=4):
    camera_mesh2 = o3d.geometry.TriangleMesh.create_cone(resolution=res)
    camera_mesh2.compute_vertex_normals()
    camera_mesh2.scale(0.25, camera_mesh2.get_center())
    camera_mesh2.translate([0, 0, 0], relative=False)

    if color:
        camera_mesh2.paint_uniform_color(color)
    return camera_mesh2


def main():
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1025)
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2021-11-22-16-30-08.json")

    image2pose = read_images()
    point3did2xyzrgb = read_points3D_coordinates()
    points_3d_list = []
    for point3d_id in point3did2xyzrgb:
        x, y, z, r, g, b = point3did2xyzrgb[point3d_id]
        points_3d_list.append([x, y, z, r/255, g/255, b/255])
    points_3d_list = np.vstack(points_3d_list)
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d_list[:, :3]))
    point_cloud.colors = o3d.utility.Vector3dVector(points_3d_list[:, 3:])
    desc_list, coord_list, im_name_list = load_2d_queries()
    point3d_id_list, point3d_desc_list = load_3d_database()
    matching_2d_to_3d(point3d_id_list, point3d_desc_list)

    cm1 = produce_cam_mesh([0.5, 1, 0.5], 20)
    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    cameras = [point_cloud, coord_mesh]
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
