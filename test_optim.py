import numpy as np
import time
import random
import pnp.build.pnp_python_binding
import open3d as o3d
from feature_matching import build_descriptors_2d, load_2d_queries_generic
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

ext_mat = np.array([[0.83, -0.05, 0.54, -3.64],
                    [0.1, 0.99, -0.04, -0.05],
                    [-0.54, 0.09, 0.83, -0.76],
                    [0, 0, 0, 1]])
f, cx, cy, k = 3031.9540853272997, 1134.0, 2016.0, 0.061174702881675876
int_mat = np.array([[f, 0, cx],
                    [0, f, cy],
                    [0, 0, 1]])
object_points = [[1.0213794597535921, -1.107638252964459, 4.007113416577096],
                 [5.641738486479504, 0.08761756228258717, 3.0840352834902034],
                 [-1.82268046636347, -1.115535316258721, 4.364550715361405],
                 [0.9404591339551345, -0.9520830071998183, 4.001604420145794],
                 [-1.6118745192959372, -0.046330574557696166, 4.573809599421738],
                 [-0.022039218268002352, -0.4603563310887896, 5.506627280603623],
                 [0.04039804635109716, -0.05590841923283404, 5.583065532557033],
                 [0.46400582097192705, 0.009715837489138148, 5.464414178585601],
                 [1.3588445344104596, -1.0682829226107526, 4.194758719096439],
                 [0.47457450678810137, -0.8516788997396209, 4.093248876229038],
                 [1.4356444064976972, -0.9347216756164156, 4.277765872588977],
                 [2.9690939890069568, -1.0538678671881005, 5.072324705190996],
                 [0.6925145475164348, -0.005994163789100908, 5.605381161715195],
                 [2.61401081408322, 1.4126709764062892, -1.0122921444918178],
                 [0.7167204971627078, -0.013303045386472619, 5.60774278228206],
                 [0.7542174207922986, 0.04134585861291182, 5.620323046819647],
                 [1.6042066463037026, -0.97496292356445, 4.385950530645061],
                 [0.28319130331987946, -1.0256946201406654, 3.5796065584691625],
                 [-0.19992388266518218, 0.11555182840554633, 5.116969186225387],
                 [3.541322728635186, 0.2511494319424901, 6.650323120486321],
                 [1.8791459342320158, -1.0978465792558905, 4.469604306841214],
                 [1.3156411741869682, -0.554331533956202, 6.321648703840549],
                 [1.3803358455669479, -0.5787503409420706, 6.309545828989181],
                 [1.3899818414467184, -0.5580425794005122, 6.342343048212614],
                 [1.4045493809478453, -0.5168465473882413, 6.329674383597053],
                 [-1.9526904519832942, 1.0823310305188905, 4.1411252721378995],
                 [2.171170533287498, 1.0279691824012591, 4.908667703575963],
                 [-1.8922658800631214, 1.0937490781098207, 4.046106791855489],
                 [1.2659081836544703, 1.119952094421854, 4.09908347868374],
                 [-2.433795099724431, 1.102270577838753, 4.325191502622594],
                 [-1.7711932096267944, 1.1010410595963884, 3.987579479806991],
                 [-1.770595947909954, 1.0963677220244452, 3.9899146881408463],
                 [-1.771165027754322, 1.0966941459559312, 4.007465427529099]]
object_points = np.array(object_points)
d1 = []
d2 = []
for xyz in object_points:
    d1.append(xyz)
    xyz2 = np.array([xyz[0], xyz[1], xyz[2], 1])
    xyz2 = ext_mat @ xyz2
    xyz2 = xyz2[:3]/xyz2[3]
    u, v, w = int_mat@xyz2
    u /= w
    v /= w
    u = (u - cx) / f
    v = (v - cy) / f
    d2.append([u, v])
d1 = np.array(d1)
d2 = np.array(d2)
res = pnp.build.pnp_python_binding.pnp(d1, d2)
print(f"pnp with correct data diff={np.sum(np.square(ext_mat-res))}")

# some noises
new_d1 = []
new_d2 = []
for i in range(d2.shape[0]):
    u, v = d2[i]
    x, y, z = d1[i]
    new_d2.append([u, v])
    new_d1.append([x, y, z])
    for _ in range(3):
        new_d2.append([u, v])
        new_d1.append([random.random()*5-5 for _ in range(3)])
print(new_d1)
print(new_d2)
new_d1 = np.array(new_d1)
new_d2 = np.array(new_d2)
res2 = pnp.build.pnp_python_binding.pnp(new_d1, new_d2)
print(f"pnp with noisy data diff={np.sum(np.square(ext_mat-res2))}")