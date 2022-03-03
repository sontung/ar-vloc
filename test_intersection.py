import open3d as o3d
import numpy as np
from pyrenderer.mathematics.shapes import TriangleSoup
from tqdm import tqdm
from taichi_glsl.randgen import randInt, rand
import taichi_glsl as ts
import taichi as ti

ti.init()
mesh = o3d.io.read_triangle_mesh("/home/sontung/work/dense_fusion_building/meshed-poisson.ply")
triangles = np.asarray(mesh.triangles)
vertices = np.asarray(mesh.vertices)
world = TriangleSoup(vertices, triangles)


@ti.kernel
def process() -> ti.i8:
    ro = ts.vec3(rand(), rand(), rand())
    rd = ts.vec3(rand(), rand(), rand())
    res = world.hit(ro, rd, 0, 1)
    return res


for _ in tqdm(range(100)):
    hit = process()
    print(hit)
