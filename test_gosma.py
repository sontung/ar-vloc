import numpy as np


ext_mat = [[0.254249, -0.346227,  0.903042, -1.256218],
           [0.189263, -0.897860, -0.397527, -1.531903],
           [0.948439,  0.271983, -0.162752, 4.787511],
           [0, 0, 0, 1]
           ]

res_line = "-1.256218195e+00 -1.531902909e+00  4.787511349e+00  2.542491555e-01  1.892632991e-01  9.484391212e-01 -3.462268710e-01 -8.978596330e-01  2.719834745e-01  9.030417204e-01 -3.975266516e-01 -1.627520323e-01 8.271529526e-02 3.937936290e-01 8.734280000e-04 1"
numbers = []
for du in res_line.split(" "):
    try:
        n = float(du)
    except ValueError:
        continue
    else:
        numbers.append(n)
trans_mat = np.identity(4).astype(np.float64)
trans_mat[0, 3] = -numbers[0]
trans_mat[1, 3] = -numbers[1]
trans_mat[2, 3] = -numbers[2]
rot_mat = np.identity(4).astype(np.float64)
rot_mat[:3, 0] = numbers[3:6]
rot_mat[:3, 1] = numbers[6:9]
rot_mat[:3, 2] = numbers[9:12]
res_mat = rot_mat@trans_mat

print(res_mat)

image_mat = np.loadtxt("debug/image.txt")
object_mat = np.loadtxt("debug/object.txt")

for idx, xyz in enumerate(object_mat):
    xyz2 = rot_mat[:3, :3]@(xyz-numbers[:3])

    xyz = np.hstack([xyz, 1])
    xyz = res_mat@xyz

    u, v, w = xyz[:3]
    u /= w
    v /= w
    uv = np.array([u, v, 1])
    test = image_mat-uv
    test = test[:, :2]
    test = np.square(test)
    test = np.sum(test, axis=1)
    if np.min(test) < 0.1:
        print(idx)
