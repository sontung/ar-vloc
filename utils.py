import open3d as o3d
import sys
import numpy as np
import trimesh
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as rot_mat_compute


def clean_point_cloud(pcd):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return cl


def clean_pc():
    sys.stdin = open("sfm_models/points3D_o3d_cleaned2.pcd", "r")
    lines = sys.stdin.readlines()
    out_buf = open("sfm_models/pc.txt", "w")
    for line in lines:
        x, y, z, _ = map(float, line.split(" ")[:7])
        print(x, y, z, file=out_buf)


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

    # ar_obj.compute_vertex_normals()
    texture = cv2.cvtColor(cv2.imread("sfm_models/texture_flip.png"), cv2.COLOR_BGR2RGB)
    ar_obj.textures = [o3d.geometry.Image(texture)]
    # ar_obj.scale(0.3, ar_obj.get_center())

    pcd = o3d.io.read_point_cloud("sfm_models/points3D_o3d_cleaned.pcd", format="pcd")
    o3d.io.write_point_cloud("sfm_models/points3D_o3d_cleaned2.pcd", pcd, write_ascii=True, compressed=True)
    bb = pcd.get_oriented_bounding_box()

    bb_points = np.asarray(bb.get_box_points())
    lines = [[0, 3]]
    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(bb_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # ar_obj.translate(bb.get_center(), relative=False)
    # ar_obj.translate([-3, 0, 1.5], relative=True)

    # ar_obj.translate([-1.5, 0, -3.15], relative=True)
    o3d.visualization.draw_geometries([ar_obj, pcd])


def simple_square(out_file="sfm_models/square.obj"):
    # default_vertices = np.array([
    #     [-0.5, -0.5, -0.5],
    #     [-0.5, -0.5, 0.5],
    #     [0.5, -0.5, 0.5],
    #     [0.5, -0.5, -0.5]
    # ])
    # default_vertices = np.array([
    #     [-1.3, -3.8, 3.018],
    #     [8.1, -3.8, 3.018],
    #     [8.1, 4.88, 3.018],
    #     [-1.3, 4.88, 3.018],
    # ])
    default_vertices = np.array(
        [[12.02863956,  0.78380664,  6.93394077],
         [ 5.39920638, -7.93330614,  7.82817919],
         [-2.9667141,  -1.18391504, 11.60110199],
         [ 3.66271909,  7.53319774, 10.70686357]]
    )
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
    mesh.show()
    trimesh.exchange.export.export_mesh(mesh, out_file, "obj")


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


if __name__ == '__main__':
    # clean_pc()
    # simple_square()
    visualize()
