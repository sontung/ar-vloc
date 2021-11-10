import open3d as o3d
import sys


def colmap2open3d(pc_file="sfm_models/points3D.txt", out_file="sfm_models/points3D_o3d.txt"):
    sys.stdin = open(pc_file, "r")
    lines = sys.stdin.readlines()
    out_buf = open(out_file, "w")
    for line in lines:
        if line[0] == "#":
            continue
        _, x, y, z, r, g, b = map(float, line.split(" ")[:7])
        print(x, y, z, r/255.0, g/255.0, b/255.0, file=out_buf)


def visualize():
    pcd = o3d.io.read_point_cloud("sfm_models/points3D_o3d.txt", format="xyzrgb")
    o3d.visualization.draw_geometries([pcd])



if __name__ == '__main__':
    visualize()