import open3d as o3d
import sys
import numpy as np
import trimesh
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as rot_mat_compute
from os import listdir
from os.path import isfile, join


def to_homo(vec):
    vec = vec.tolist()
    vec.append(1.0)
    return np.array(vec)


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


# @profile
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    # v1_u = unit_vector(v1)
    # v2_u = unit_vector(v2)
    dot_p = np.dot(v1, v2)
    clip_dot = max(-1, min(dot_p, 1))
    return np.arccos(clip_dot)


def rewrite_colmap_output(in_dir, out_dir):
    """
    add an prefix to colmap database images' names
    """
    sys.stdin = open(in_dir, "r")
    lines = sys.stdin.readlines()
    idx = 0
    lines_to_be_written = []

    while idx < len(lines):
        line = lines[idx]
        if line[0] == "#":
            idx += 1
            lines_to_be_written.append(line[:-1])
            continue
        else:
            image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, image_name = line[:-1].split(" ")
            image_name = f"db/{image_name}"
            new_line = " ".join([image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, image_name])
            lines_to_be_written.append(new_line)
            lines_to_be_written.append(lines[idx + 1][:-1])
            idx += 2
    with open(out_dir, "w") as a_file:
        for line in lines_to_be_written:
            print(line, file=a_file)


def rewrite_retrieval_output(in_dir, out_dir):
    sys.stdin = open(in_dir, "r")
    lines = sys.stdin.readlines()
    lines_to_be_written = []
    for line in lines:
        tokens = line.split(" ")
        tokens = [token.rstrip().split("/")[-1] for token in tokens]
        line2 = " ".join(tokens)
        lines_to_be_written.append(line2)
    with open(out_dir, "w") as a_file:
        for line in lines_to_be_written:
            print(line, file=a_file)


def write_something(out_dir):
    with open(out_dir, "w") as a_file:
        for idx in range(30):
            print(f"line-{idx}.jpg", file=a_file)


def read_videos():
    mypath = "/home/sontung/work/recon_models/building/videos"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for afile in onlyfiles:
        afile = f"{mypath}/{afile}"
        name = afile.split("/")[-1].split(".")[-2]
        cap = cv2.VideoCapture(afile)
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                idx += 1
                if idx % 4 == 0:
                    cv2.imwrite(f"/home/sontung/work/recon_models/building/images/img-{name}-{idx}.jpg", frame)
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


def read_video(afile, save_folder):
    name = afile.split("/")[-1].split(".")[-2]
    cap = cv2.VideoCapture(afile)
    idx = 0
    img_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            idx += 1
            if idx % 4 == 0:
                img_idx += 1
                cv2.imwrite(f"{save_folder}/image{img_idx:04d}.jpg", frame)
                print(img_idx)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def create_image_list():
    from os import listdir
    from os.path import isfile, join
    mypath = "/home/sontung/work/recon_models/building/images"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    k2name = {}
    for name in onlyfiles:
        k = name.split("-")[1]
        if k not in k2name:
            k2name[k] = [name]
        else:
            k2name[k].append(name)
    for k in k2name:
        txt_file = open(f"/home/sontung/work/recon_models/building/{k}-img-list.txt", "w")
        for name in k2name[k]:
            print(name, file=txt_file)
        txt_file.close()


def dump_point_cloud(file_dir="/home/sontung/work/recon_models/indoor/model.txt"):
    from colmap_io import read_points3D_coordinates
    data = read_points3D_coordinates("/home/sontung/work/recon_models/indoor/dense_model/points3D.txt")
    afile = open(file_dir, "w")
    for k in data:
        x, y, z, r, g, b = data[k]
        print(x, y, z, r, g, b, file=afile)
    return


if __name__ == '__main__':
    rewrite_retrieval_output("/home/sontung/work/Hierarchical-Localization/outputs/hblab/pairs-query-netvlad20.txt",
                             "data/retrieval_pairs.txt")
    # write_something("vloc_workspace_retrieval/test_images.txt")
    # rewrite_colmap_output("/home/sontung/work/Hierarchical-Localization/outputs/hblab/sfm_sift/images.txt",
    #                       "/home/sontung/work/Hierarchical-Localization/outputs/hblab/sfm_sift/images2.txt")
    # dump_point_cloud()
    # clean_pc()
    # simple_square()
    # read_video("/media/sontung/580ECE740ECE4B28/recon_models2/indoor2/IMG_0794.MOV",
    #            "/media/sontung/580ECE740ECE4B28/recon_models2/indoor2/images")
    # create_image_list()
    # read_video("/home/sontung/work/recon_models/building/videos/c4.MOV")
