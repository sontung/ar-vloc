import sys
import numpy as np
import time
import cv2
import open3d as o3d
from feature_matching import build_descriptors_2d, load_2d_queries_opencv, load_2d_queries_generic
from colmap_io import read_points3D_coordinates, read_images, read_cameras
from scipy.spatial import KDTree
from point3d import PointCloud, Point3D
from point2d import FeatureCloud
from vocab_tree import VocabTree
from PIL import Image


DEBUG_2D_3D_MATCHING = True
MATCHING_BENCHMARK = True


def evaluate(res, tree_coord, point3d_cloud, ref_3d_id):
    count, samples = 0, 0
    for feature, point in res:
        dist, closest_feature = tree_coord.query(feature.xy)
        if dist < 2:
            ref_point = point3d_cloud.access_by_id(ref_3d_id[closest_feature])
            samples += 1
            if ref_point == point:
                count += 1
    return count, samples


def visualize_matching_helper(query_image, feature, point, sfm_image_folder):
    visualized_list = []
    (x, y) = map(int, feature.xy)
    cv2.circle(query_image, (x, y), 50, (128, 128, 0), -1)
    cv2.circle(query_image, (y, x), 50, (255, 128, 0), -1)

    visualized_list.append(query_image)
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


def visualize_matching(bf_results, results, query_image_ori, sfm_image_folder):
    for ind in range(len(results)):
        print(results[ind][0].xy, bf_results[ind][0].xy)
        assert np.sum(results[ind][0].xy-bf_results[ind][0].xy) < 0.1
        feature, point, dist = results[ind]
        query_image = np.copy(query_image_ori)
        l1 = visualize_matching_helper(query_image, feature, point, sfm_image_folder)
        # cv2.imshow("t", l1)

        feature, point, dist = bf_results[ind]
        if point is not None:
            query_image = np.copy(query_image_ori)
            l2 = visualize_matching_helper(query_image, feature, point, sfm_image_folder)
            # cv2.imshow("t2", l2)
        else:
            print("unreliable match")

        # cv2.waitKey()
        # cv2.destroyAllWindows()


def main():
    query_images_folder = "Test line small"
    sfm_images_dir = "sfm_ws_hblab/images.txt"
    sfm_point_cloud_dir = "sfm_ws_hblab/points3D.txt"
    sfm_images_folder = "sfm_ws_hblab/images"

    # query_images_folder = "test_images"
    # sfm_images_dir = "sfm_models/images.txt"
    # sfm_point_cloud_dir = "sfm_models/points3D.txt"
    # sfm_images_folder = "sfm_models/images"

    image2pose = read_images(sfm_images_dir)
    point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)
    point3d_id_list, point3d_desc_list, point3did2descs = build_descriptors_2d(image2pose, sfm_images_folder)

    point3d_cloud = PointCloud(point3did2descs, debug=MATCHING_BENCHMARK)
    for i in range(len(point3d_id_list)):
        point3d_id = point3d_id_list[i]
        point3d_desc = point3d_desc_list[i]
        xyzrgb = point3did2xyzrgb[point3d_id]
        point3d_cloud.add_point(point3d_id, point3d_desc, xyzrgb[:3], xyzrgb[3:])
    point3d_cloud.commit()
    vocab_tree = VocabTree(point3d_cloud)
    gt_data = {}
    for image in image2pose:
        if image not in image2pose:
            continue
        data = image2pose[image]
        ref_coords = []
        ref_3d_id = []
        for x, y, pid in data[1]:
            if pid > 0:
                ref_coords.append([x, y])
                ref_3d_id.append(pid)
                # print(pid)
                a_point = point3d_cloud.access_by_id(pid)
                if a_point is not None:
                    a_point.assign_visibility(data[0], (x, y))
        gt_data[data[0]] = (ref_coords, ref_3d_id)

    desc_list, coord_list, im_name_list, _, image_list = load_2d_queries_generic(query_images_folder)
    p2d2p3d = {}
    start_time = time.time()
    count_all = 0
    samples_all = 0
    vocab_based = [0, 0, 0]
    active_search_based = [0, 0, 0]
    for i in range(len(desc_list)):
        print(f"Matching {i}/{len(desc_list)}")
        point2d_cloud = FeatureCloud()
        if im_name_list[0] in gt_data:
            ref_coords, ref_3d_id = gt_data[im_name_list[0]]
        else:
            ref_coords, ref_3d_id = [], []

        for j in range(coord_list[i].shape[0]):
            point2d_cloud.add_point(i, desc_list[i][j], coord_list[i][j])
        point2d_cloud.assign_words(vocab_tree.word2level, vocab_tree.v1)

        start = time.time()
        res, count, samples, bf_res = vocab_tree.search(point2d_cloud, nb_matches=100, debug=True)
        print(f"Accuracy compared to brute force {count}/{samples}")
        vocab_based[0] += time.time() - start
        if DEBUG_2D_3D_MATCHING:
            visualize_matching(bf_res, res, image_list[i], sfm_images_folder)
        if len(ref_3d_id) > 0:
            tree_coord = KDTree(np.array(ref_coords))
            count, samples = evaluate(res, tree_coord, point3d_cloud, ref_3d_id)

        sys.exit()

        vocab_based[1] += count
        vocab_based[2] += samples

        start = time.time()
        res, count, samples = vocab_tree.active_search(point2d_cloud, nb_matches=100)
        active_search_based[0] += time.time() - start

        count, samples = evaluate(res, tree_coord, point3d_cloud, ref_3d_id)
        active_search_based[1] += count
        active_search_based[2] += samples

        count_all += count
        samples_all += samples

        p2d2p3d[i] = []
        for point2d, point3d in res:
            p2d2p3d[i].append((point2d.xy, point3d.xyz))

    time_spent = time.time() - start_time
    print(f"Matching 2D-3D done in {round(time_spent, 3)} seconds, "
          f"avg. {round(time_spent / len(desc_list), 3)} seconds/image")
    if MATCHING_BENCHMARK:
        print("vocab", vocab_based[0], vocab_based[1] / vocab_based[2])
        print("active", active_search_based[0], active_search_based[1] / active_search_based[2])

        print(f"Matching accuracy={round(count_all / samples_all * 100, 3)}%")



if __name__ == '__main__':
    main()
