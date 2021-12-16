import numpy as np
import time
import open3d as o3d
from feature_matching import build_descriptors_2d, load_2d_queries_generic
from colmap_io import read_points3D_coordinates, read_images, read_cameras
from colmap_read import qvec2rotmat
from localization import localize_single_image
from scipy.spatial import KDTree
from vis_utils import produce_sphere, produce_cam_mesh, visualize_2d_3d_matching_single
from PIL import Image
from active_search import matching_active_search
from point3d import PointCloud, Point3D
from point2d import FeatureCloud
from vocab_tree import VocabTree


VISUALIZING_SFM_POSES = False
VISUALIZING_POSES = True
BRUTE_FORCE_MATCHING = True


def move_cam(result, color):
    if result is None:
        return None
    rot_mat, trans = result
    t = -rot_mat.transpose() @ trans
    t = t.reshape((3, 1))
    mat = np.hstack([-rot_mat.transpose(), t])
    mat = np.vstack([mat, np.array([0, 0, 0, 1])])

    cm = produce_cam_mesh(color=color)

    vertices = np.asarray(cm.vertices)
    for i in range(vertices.shape[0]):
        arr = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1])
        arr = mat @ arr
        vertices[i] = arr[:3]
    cm.vertices = o3d.utility.Vector3dVector(vertices)
    return cm


def main():
    query_images_folder = "Test line"
    cam_info_dir = "sfm_ws_hblab/cameras.txt"
    sfm_images_dir = "sfm_ws_hblab/images.txt"
    sfm_point_cloud_dir = "sfm_ws_hblab/points3D.txt"
    sfm_images_folder = "sfm_ws_hblab/images"

    # query_images_folder = "test_images"
    # cam_info_dir = "sfm_models/cameras.txt"
    # sfm_images_dir = "sfm_models/images.txt"
    # sfm_point_cloud_dir = "sfm_models/points3D.txt"
    # sfm_images_folder = "sfm_models/images"

    image2pose = read_images(sfm_images_dir)
    point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)
    points_3d_list = []
    point3d_id_list, point3d_desc_list, p3d_desc_list_multiple, point3did2descs = build_descriptors_2d(image2pose, sfm_images_folder)

    point3d_cloud = PointCloud(point3did2descs)
    for i in range(len(point3d_id_list)):
        point3d_id = point3d_id_list[i]
        point3d_desc = point3d_desc_list[i]
        xyzrgb = point3did2xyzrgb[point3d_id]
        point3d_cloud.add_point(point3d_id, point3d_desc, p3d_desc_list_multiple[i], xyzrgb[:3], xyzrgb[3:])
    point3d_cloud.commit(image2pose)
    if BRUTE_FORCE_MATCHING:
        point3d_cloud.build_desc_tree()
    vocab_tree = VocabTree(point3d_cloud)

    if VISUALIZING_POSES:
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

    desc_list, coord_list, im_name_list, metadata_list, image_list, response_list = load_2d_queries_generic(query_images_folder)
    p2d2p3d = {}
    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    for i in range(len(desc_list)):
        print(f"Matching {i+1}/{len(desc_list)}")
        point2d_cloud = FeatureCloud()
        for j in range(coord_list[i].shape[0]):
            point2d_cloud.add_point(j, desc_list[i][j], coord_list[i][j], response_list[i][j])
        point2d_cloud.assign_words(vocab_tree.word2level, vocab_tree.v1)

        # res, _, _ = vocab_tree.active_search(point2d_cloud)
        # res, _, _ = vocab_tree.search(point2d_cloud)
        res = vocab_tree.search_experimental(point2d_cloud, image_list[i],
                                             sfm_images_folder, nb_matches=30)

        p2d2p3d[i] = []
        if len(res[0]) > 2:
            for point2d, point3d, _ in res:
                p2d2p3d[i].append((point2d.xy, point3d.xyz))
        else:
            for point2d, point3d in res:
                p2d2p3d[i].append((point2d.xy, point3d.xyz))

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1025)

        metadata = metadata_list[i]
        if len(metadata) == 0:
            pass
        print(f"Localizing image {im_name_list[i]}")

        f = metadata["f"]*100
        cx = metadata["cx"]
        cy = metadata["cy"]
        k = 0.06

        camera_matrix = np.array([[f, 0, 0],
                                  [0, f, 0],
                                  [0, 0, -1]])
        distortion_coefficients = np.array([k, 0, 0, 0])
        result = localize_single_image(p2d2p3d[i], camera_matrix, distortion_coefficients)
        cm = move_cam(result, [0, 1, 0])
        if cm is not None:
            vis.add_geometry(cm)

        camera_matrix = np.array([[f, 0, cx],
                                  [0, f, cy],
                                  [0, 0, 1]])
        distortion_coefficients = np.array([k, 0, 0, 0])
        result = localize_single_image(p2d2p3d[i], camera_matrix, distortion_coefficients)
        cm = move_cam(result, [0, 0, 1])
        if cm is not None:
            vis.add_geometry(cm)

        vis.add_geometry(point_cloud)
        vis.add_geometry(coord_mesh)
        vis.run()
        vis.destroy_window()


if __name__ == '__main__':
    main()
