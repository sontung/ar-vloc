import sys
import numpy as np
import cv2
import random
import psutil
import time
import open3d as o3d
from feature_matching import build_descriptors_2d, load_2d_queries_opencv
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


NEXT = False
DEBUG_2D_3D_MATCHING = False
DEBUG_PNP = False
VISUALIZING_SFM_POSES = True
VISUALIZING_POSES = False
MATCHING_BENCHMARK = True


def matching_2d_to_3d(point3d_id_list, point3d_desc_list, point2d_desc_list):
    """
    returns [image id] => point 2d id => point 3d id
    """
    start_time = time.time()
    kd_tree = KDTree(point3d_desc_list)
    print("Done constructing K-D tree")
    result = {i: [] for i in range(len(point2d_desc_list))}
    for i in range(len(point2d_desc_list)):
        desc_list = point2d_desc_list[i]
        for j in range(desc_list.shape[0]):
            desc = desc_list[j]
            res = kd_tree.query(desc, 2)
            if res[0][1] > 0.0:
                if res[0][0]/res[0][1] < 0.7:  # ratio test
                    result[i].append([j, point3d_id_list[res[1][0]]])
    time_spent = time.time()-start_time
    print(f"Matching 2D-3D done in {round(time_spent, 3)} seconds, "
          f"avg. {round(time_spent/len(point2d_desc_list), 3)} seconds/image")
    return result


def key_sw(u):
    global NEXT
    NEXT = not NEXT


def visualize_2d_3d_matching(p2d2p3d, coord_2d_list, im_name_list, point3did2xyzrgb,
                             original_point_cloud):
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


def produce_name2pose(image2pose):
    res = {image2pose[k][0]: [image2pose[k][2], image2pose[k][3]] for k in image2pose}
    return res


def main():
    query_images_folder = "sfm_ws_hblab/images"
    cam_info_dir = "sfm_ws_hblab/cameras.txt"
    sfm_images_dir = "sfm_ws_hblab/images.txt"
    sfm_point_cloud_dir = "sfm_ws_hblab/points3D.txt"
    sfm_images_folder = "sfm_ws_hblab/images"

    # query_images_folder = "test_images"
    # cam_info_dir = "sfm_models/cameras.txt"
    # sfm_images_dir = "sfm_models/images.txt"
    # sfm_point_cloud_dir = "sfm_models/points3D.txt"
    # sfm_images_folder = "sfm_models/images"

    camid2params = read_cameras(cam_info_dir)
    image2pose = read_images(sfm_images_dir)
    name2pose = produce_name2pose(image2pose)
    point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)
    points_3d_list = []
    point3d_id_list, point3d_desc_list = build_descriptors_2d(image2pose, sfm_images_folder)

    point3d_cloud = PointCloud(debug=MATCHING_BENCHMARK)
    for i in range(len(point3d_id_list)):
        point3d_id = point3d_id_list[i]
        point3d_desc = point3d_desc_list[i]
        xyzrgb = point3did2xyzrgb[point3d_id]
        point3d_cloud.add_point(point3d_id, point3d_desc, xyzrgb[:3], xyzrgb[3:])
    point3d_cloud.commit()
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

    desc_list, coord_list, im_name_list = load_2d_queries_opencv(query_images_folder)
    p2d2p3d = {}
    start_time = time.time()
    count_all = 0
    samples_all = 0
    vocab_based = [0, 0, 0]
    active_search_based = [0, 0, 0]
    for i in range(len(desc_list)):
        point2d_cloud = FeatureCloud()
        for j in range(coord_list[i].shape[0]):
            point2d_cloud.add_point(i, desc_list[i][j], coord_list[i][j])
        point2d_cloud.assign_words(vocab_tree.word2level, vocab_tree.v1)

        start = time.time()
        res, count, samples = point3d_cloud.matching_2d_to_3d_vocab_based(point2d_cloud,
                                                                          debug=MATCHING_BENCHMARK)

        vocab_based[0] += time.time()-start
        vocab_based[1] += count
        vocab_based[2] += samples

        start = time.time()
        res, count, samples = vocab_tree.search(point2d_cloud, debug=MATCHING_BENCHMARK)
        active_search_based[0] += time.time()-start
        active_search_based[1] += count
        active_search_based[2] += samples

        count_all += count
        samples_all += samples

        p2d2p3d[i] = []
        for point2d, point3d in res:
            p2d2p3d[i].append((point2d.xy, point3d.xyz))
    time_spent = time.time()-start_time
    print(f"Matching 2D-3D done in {round(time_spent, 3)} seconds, "
          f"avg. {round(time_spent/len(desc_list), 3)} seconds/image")
    if MATCHING_BENCHMARK:
        print("vocab", vocab_based[0], vocab_based[1]/vocab_based[2])
        print("active", active_search_based[0], active_search_based[1]/active_search_based[2])

        print(f"Matching accuracy={round(count_all/samples_all*100, 3)}%")

    localization_results = []

    if DEBUG_PNP:
        index = -1
        total_rot_error = 0
        total_trans_error = 0

        for im_idx in p2d2p3d:
            index += 1
            im_name = im_name_list[im_idx]
            ref_cam_pose, cam_id = name2pose[im_name]
            cam_model, _, _, cam_params = camid2params[cam_id]

            if cam_model != "SIMPLE_RADIAL":
                print("camera model is not known, skipping")
                continue

            f, cx, cy, k = cam_params
            camera_matrix = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            distortion_coefficients = np.array([k, 0, 0, 0])
            res = localize_single_image(p2d2p3d[im_idx], camera_matrix, distortion_coefficients)

            if res is None:
                continue
            rot_mat, trans = res
            localization_results.append(res)
            qw, qx, qy, qz, tx, ty, tz = ref_cam_pose
            ref_rot_mat = qvec2rotmat([qw, qx, qy, qz])
            rot_error = np.sum(np.abs(ref_rot_mat-rot_mat))
            trans_error = np.sum(np.abs(trans-np.array([[tx], [ty], [tz]])))
            total_rot_error += rot_error
            total_trans_error += trans_error
        print(f"total rotation error={round(total_rot_error, 4)}, total translation error={round(total_trans_error, 4)}")

    if VISUALIZING_POSES:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1025)

        point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        cameras = [point_cloud, coord_mesh]

        # ground-truth poses from sfm
        if VISUALIZING_SFM_POSES:
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


if __name__ == '__main__':
    main()
