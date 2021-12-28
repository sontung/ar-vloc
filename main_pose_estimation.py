import numpy as np
import time
import open3d as o3d
from feature_matching import build_descriptors_2d, load_2d_queries_generic
from colmap_io import read_points3D_coordinates, read_images, read_cameras
from colmap_read import qvec2rotmat
from localization import localize_single_image, localize_single_image_lt_pnp
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


def main():
    query_images_folder = "Test line"
    sfm_images_dir = "sfm_ws_hblab/images.txt"
    sfm_point_cloud_dir = "sfm_ws_hblab/points3D.txt"
    sfm_images_folder = "sfm_ws_hblab/images"

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
    vocab_tree.load_matching_pairs(query_images_folder)

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
    start_time = time.time()
    for i in range(len(desc_list)):
        print(f"Matching {i+1}/{len(desc_list)}: {im_name_list[i]}")
        point2d_cloud = FeatureCloud()
        for j in range(coord_list[i].shape[0]):
            point2d_cloud.add_point(j, desc_list[i][j], coord_list[i][j], response_list[i][j])
        point2d_cloud.assign_words(vocab_tree.word2level, vocab_tree.v1)

        res = vocab_tree.search_brute_force(point2d_cloud, im_name_list[i], query_images_folder)

        p2d2p3d[i] = []
        if len(res[0]) > 2:
            for point2d, point3d, _ in res:
                p2d2p3d[i].append((point2d.xy, point3d.xyz))
        else:
            for point2d, point3d in res:
                p2d2p3d[i].append((point2d.xy, point3d.xyz))

    time_spent = time.time()-start_time
    print(f"Matching 2D-3D done in {round(time_spent, 3)} seconds, "
          f"avg. {round(time_spent/len(desc_list), 3)} seconds/image")

    localization_results = []
    for im_idx in p2d2p3d:
        print(f"Localizing image {im_name_list[im_idx]}")
        metadata = metadata_list[im_idx]
        f = metadata["f"]*100
        cx = metadata["cx"]
        cy = metadata["cy"]
        k = 0.06
        # f, cx, cy, k = 3031.9540853272997, 1134.0, 2016.0, 0.061174702881675876
        camera_matrix = np.array([[f, 0, cx],
                                  [0, f, cy],
                                  [0, 0, 1]])
        distortion_coefficients = np.array([k, 0, 0, 0])
        res = localize_single_image(p2d2p3d[im_idx], camera_matrix, distortion_coefficients)
        res2 = localize_single_image_lt_pnp(p2d2p3d[im_idx], f, cx, cy)

        if res is None:
            continue
        localization_results.append((res, "vector"))
        localization_results.append((res2, "vector2"))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1025)

    point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    cameras = [point_cloud, coord_mesh]
    for c in cameras:
        vis.add_geometry(c)

    # queried poses
    for result in localization_results:
        cm = None
        if result is None:
            continue
        rot_mat, trans = result[0]
        t = -rot_mat.transpose() @ trans
        t = t.reshape((3, 1))
        mat = np.hstack([-rot_mat.transpose(), t])
        mat = np.vstack([mat, np.array([0, 0, 0, 1])])
        if result[-1] == "vector":
            cm = produce_cam_mesh(color=[0, 1, 0])

        elif result[-1] == "vector2":
            cm = produce_cam_mesh(color=[0, 0, 1])

        vertices = np.asarray(cm.vertices)
        for i in range(vertices.shape[0]):
            arr = np.array([vertices[i, 0], vertices[i, 1], vertices[i, 2], 1])
            arr = mat @ arr
            vertices[i] = arr[:3]
        cm.vertices = o3d.utility.Vector3dVector(vertices)
        cameras.append(cm)
        vis.add_geometry(cm, reset_bounding_box=False)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
