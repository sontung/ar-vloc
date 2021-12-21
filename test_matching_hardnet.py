import sys
import numpy as np
import time
import cv2
import os
import glob
import open3d as o3d
from hardnet import build_descriptors_2d as build_descriptors_2d_hardnet
from hardnet import load_2d_queries_generic as load_2d_queries_generic_hardnet
from feature_matching import build_descriptors_2d, load_2d_queries_opencv, load_2d_queries_generic
from colmap_io import read_points3D_coordinates, read_images
from scipy.spatial import KDTree
from point3d import PointCloud, Point3D
from point2d import FeatureCloud
from vocab_tree import VocabTree
from PIL import Image
from vis_utils import visualize_matching_and_save, visualize_matching


DEBUG_2D_3D_MATCHING = True
MATCHING_BENCHMARK = True


def main():
    using_ps = True
    query_images_folder = "Test line small"
    sfm_images_dir = "sfm_ws_hblab/images.txt"
    sfm_point_cloud_dir = "sfm_ws_hblab/points3D.txt"
    sfm_images_folder = "sfm_ws_hblab/images"

    image2pose = read_images(sfm_images_dir)
    point3did2xyzrgb = read_points3D_coordinates(sfm_point_cloud_dir)
    read_data = build_descriptors_2d_hardnet(image2pose,
                                             sfm_images_folder,
                                             using_ps=using_ps)
    point3d_id_list, point3d_desc_list, p3d_desc_list_multiple, point3did2descs = read_data

    point3d_cloud = PointCloud(point3did2descs, debug=MATCHING_BENCHMARK)
    for i in range(len(point3d_id_list)):
        point3d_id = point3d_id_list[i]
        point3d_desc = point3d_desc_list[i]
        xyzrgb = point3did2xyzrgb[point3d_id]
        point3d_cloud.add_point(point3d_id, point3d_desc, p3d_desc_list_multiple[i], xyzrgb[:3], xyzrgb[3:])

    point3d_cloud.commit(image2pose)
    vocab_tree = VocabTree(point3d_cloud, debug=True)
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
                a_point = point3d_cloud.access_by_id(pid)
                if a_point is not None:
                    a_point.assign_visibility(data[0], (x, y))
        gt_data[data[0]] = (ref_coords, ref_3d_id)

    read_queries = load_2d_queries_generic_hardnet(query_images_folder, using_ps=using_ps)
    desc_list, coord_list, im_name_list, _, image_list, response_list = read_queries
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
            point2d_cloud.add_point(j, desc_list[i][j], coord_list[i][j], response_list[i][j])
        point2d_cloud.assign_words(vocab_tree.word2level, vocab_tree.v1)

        start = time.time()
        res = vocab_tree.search_experimental(point2d_cloud, image_list[i],
                                             sfm_images_folder, nb_matches=100)


if __name__ == '__main__':
    # dump_matches()
    main()
