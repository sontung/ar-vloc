import numpy as np
import open3d as o3d
from feature_matching import build_descriptors_2d, load_2d_queries_generic
from colmap_io import read_points3D_coordinates, read_images, read_cameras
from localization import localize_single_image
from vis_utils import produce_sphere, produce_cam_mesh, visualize_2d_3d_matching_single
from point3d import PointCloud, Point3D
from point2d import FeatureCloud
from vocab_tree import VocabTree


VISUALIZING_SFM_POSES = False
VISUALIZING_POSES = True


query_images_folder = "Test line small"
sfm_images_dir = "sfm_ws_hblab/images.txt"
sfm_point_cloud_dir = "sfm_ws_hblab/points3D.txt"
sfm_images_folder = "sfm_ws_hblab/images"

image2pose = read_images(sfm_images_dir)
point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)
points_3d_list = []
point3d_id_list, point3d_desc_list, p3d_desc_list_multiple, point3did2descs = build_descriptors_2d(image2pose,
                                                                                                   sfm_images_folder)

point3d_cloud = PointCloud(point3did2descs)
for i in range(len(point3d_id_list)):
    point3d_id = point3d_id_list[i]
    point3d_desc = point3d_desc_list[i]
    xyzrgb = point3did2xyzrgb[point3d_id]
    point3d_cloud.add_point(point3d_id, point3d_desc, p3d_desc_list_multiple[i], xyzrgb[:3], xyzrgb[3:])
point3d_cloud.commit(image2pose)
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

desc_list, coord_list, im_name_list, metadata_list, _ = load_2d_queries_generic(query_images_folder)
p2d2p3d = {}
for i in range(len(desc_list)):
    point2d_cloud = FeatureCloud()
    for j in range(coord_list[i].shape[0]):
        point2d_cloud.add_point(i, desc_list[i][j], coord_list[i][j])
    point2d_cloud.assign_words(vocab_tree.word2level, vocab_tree.v1)

    # res, _, _ = vocab_tree.search(point2d_cloud)
    res, _, _ = vocab_tree.search_brute_force(point2d_cloud)

    p2d2p3d[i] = []
    if len(res[0]) > 2:
        for count, (point2d, point3d, _) in enumerate(res):
            p2d2p3d[i].append((point2d.xy, point3d.xyz))
    else:
        for point2d, point3d in res:
            p2d2p3d[i].append((point2d.xy, point3d.xyz))

localization_results = []
for im_idx in p2d2p3d:
    metadata = metadata_list[im_idx]
    if len(metadata) == 0:
        pass
    f = metadata["f"]*100
    cx = metadata["cx"]
    cy = metadata["cy"]
    k = 0.06
    print(f, cx, cy)
    # f, cx, cy, k = 3031.9540853272997, 1134.0, 2016.0, 0.061174702881675876
    camera_matrix = np.array([[f, 0, cx],
                              [0, f, cy],
                              [0, 0, 1]])
    distortion_coefficients = np.array([k, 0, 0, 0])
    res = localize_single_image(p2d2p3d[im_idx], camera_matrix, distortion_coefficients)

    if res is None:
        continue
    localization_results.append(res)

vis = o3d.visualization.Visualizer()
vis.create_window(width=1920, height=1025)

point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
cameras = [point_cloud, coord_mesh]

# queried poses
for result in localization_results:
    if result is None:
        continue
    rot_mat, trans = result
    t = -rot_mat.transpose() @ trans
    t = t.reshape((3, 1))
    mat = np.hstack([-rot_mat.transpose(), t])
    mat = np.vstack([mat, np.array([0, 0, 0, 1])])

    cm = produce_cam_mesh(color=[0, 1, 0])

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

