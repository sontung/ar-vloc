import sys
import cv2
import numpy as np


def read_points3D(in_dir="sfm_models/points3D.txt"):
    sys.stdin = open(in_dir, "r")
    lines = sys.stdin.readlines()
    data = {}
    for line in lines:
        if line[0] == "#":
            continue
        numbers = line[:-1].split(" ")
        numbers = list(map(float, numbers))
        point3d_id, x, y, z, r, g, b = numbers[:7]
        tracks = list(map(int, numbers[8:]))
        point3d_id = int(point3d_id)
        data[point3d_id] = [tracks]
    return data


def read_images(in_dir="sfm_models/images.txt"):
    sys.stdin = open(in_dir, "r")
    lines = sys.stdin.readlines()
    data = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line[0] == "#":
            idx += 1
            continue
        else:
            image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, image_name = line[:-1].split(" ")
            cam_pose = list(map(float, [qw, qx, qy, qz, tx, ty, tz]))
            image_id, cam_id = list(map(int, [image_id, cam_id]))
            points2d = list(map(float, lines[idx+1][:-1].split(" ")))
            points2d_meaningful = []  # [x, y, point 3d id]
            for i in range(0, len(points2d), 3):
                point = (points2d[i], points2d[i+1], int(points2d[i+2]))
                points2d_meaningful.append(point)

            data[image_id] = [image_name, points2d_meaningful, cam_pose, cam_id]
            idx += 2
    return data


def visualize_matching_pairs():
    point3d = read_points3D()
    images = read_images()
    images_mat = [cv2.imread(f"sfm_models/images/{images[u][0]}") for u in images]
    track_lengths = []
    for point3d_id in point3d:
        tracks = point3d[point3d_id][0]
        images_to_visualized = []
        if len(tracks) <= 10:
            continue
        for i in range(0, len(tracks), 2):

            image_id = tracks[i]
            point2d_id = tracks[i+1]

            image = images_mat[image_id-1].copy()
            px, py, point3d_id2 = images[image_id][1][point2d_id]
            # point3d_ids = [u[-1] for u in images[image_id][1] if u[-1] > 0]
            cv2.circle(image, (int(px), int(py)), 20, (255, 0, 0), -1)

            images_to_visualized.append(image)
        final_image = np.hstack(images_to_visualized)
        final_image = cv2.resize(final_image, (final_image.shape[1]//4,
                                               final_image.shape[0]//4))
        cv2.imshow("t", final_image)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    visualize_matching_pairs()
