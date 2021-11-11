import open3d as o3d
import sys
import numpy as np
import trimesh
import cv2
from PIL import Image


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
    ar_obj = o3d.io.read_triangle_mesh('sfm_models/square.obj')
    ar_obj.compute_vertex_normals()
    texture = cv2.cvtColor(cv2.imread("sfm_models/texture.jpg"), cv2.COLOR_BGR2RGB)
    ar_obj.textures = [o3d.geometry.Image(texture)]

    pcd = o3d.io.read_point_cloud("sfm_models/points3D_o3d.txt", format="xyzrgb")
    o3d.visualization.draw_geometries([ar_obj])


def simple_square(out_file="sfm_models/square.obj"):
    default_vertices = np.array([
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, -0.5, -0.5]
    ])
    default_faces = np.array([
        [2, 1, 0],
        [0, 3, 2]
    ], np.uint)

    texture = trimesh.visual.texture.TextureVisuals(uv=np.array([[0.0, 0.0],
                                                                 [0.0, 1.0],
                                                                 [1.0, 1.0],
                                                                 [1.0, 0.0]]),
                                                    image=Image.open("sfm_models/texture.jpg"))
    mesh = trimesh.Trimesh(vertices=default_vertices,
                           faces=default_faces,
                           process=False, visual=texture)
    # mesh.show()
    trimesh.exchange.export.export_mesh(mesh, out_file, "obj")


if __name__ == '__main__':
    # simple_square()
    visualize()