import open3d as o3d


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


pcd = o3d.io.read_point_cloud("sfm_models/points3D_o3d.txt", format="xyzrgb")
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                         std_ratio=1.0)
o3d.io.write_point_cloud("sfm_models/points3D_o3d_cleaned.pcd", cl)
