import sys
from typing import Dict
from numpy.lib.stride_tricks import sliding_window_view
import cv2
import os
import pyheif
import numpy as np
from tqdm import tqdm
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import kornia
import exifread
from pathlib import Path
from PIL import Image


class HardNet(nn.Module):
    patch_size = 32

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        # use torch.hub to load pretrained model
        if pretrained:
            url_link = "https://github.com/DagnyT/hardnet/raw/master/pretrained/pretrained_all_datasets/HardNet++.pth"
            pretrained_dict = torch.hub.load_state_dict_from_url(
                url_link, map_location=lambda storage, loc: storage
            )
            self.load_state_dict(pretrained_dict['state_dict'], strict=True)
        self.eval()

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Utility function that normalizes the input by batch."""
        sp, mp = torch.std_mean(x, dim=(-3, -2, -1), keepdim=True)
        # WARNING: we need to .detach() input, otherwise the gradients produced by
        # the patches extractor with F.grid_sample are very noisy, making the detector
        # training totally unstable.
        return (x - mp.detach()) / (sp.detach() + eps)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        x_norm: torch.Tensor = self._normalize_input(_input)
        x_features: torch.Tensor = self.features(x_norm)
        x_out = x_features.view(x_features.size(0), -1)
        return F.normalize(x_out, dim=1)


class HardNet2(nn.Module):

    patch_size = 32

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=8, bias=True)
        )

        # use torch.hub to load pretrained model
        if pretrained:
            self.load_state_dict(torch.load("data/HardNetPS.pth"), strict=True)
        self.eval()

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Utility function that normalizes the input by batch."""
        sp, mp = torch.std_mean(x, dim=(-3, -2, -1), keepdim=True)
        # WARNING: we need to .detach() input, otherwise the gradients produced by
        # the patches extractor with F.grid_sample are very noisy, making the detector
        # training totally unstable.
        return (x - mp.detach()) / (sp.detach() + eps)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        x_norm: torch.Tensor = self._normalize_input(_input)
        x_features: torch.Tensor = self.features(x_norm)
        x_out = x_features.view(x_features.size(0), -1)
        return F.normalize(x_out, dim=1)


def load_2d_queries_generic(folder, using_ps):
    im_names = os.listdir(folder)
    descriptors = []
    coordinates = []
    md_list = []
    im_list = []
    response_list = []
    for name in tqdm(im_names, desc="Reading query images"):
        metadata = {}
        im_name = os.path.join(folder, name)
        if "HEIC" in name:
            heif_file = pyheif.read(im_name)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            im = np.array(image)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            f = open(im_name, 'rb')
            tags = exifread.process_file(f)
            metadata["f"] = float(tags["EXIF FocalLengthIn35mmFilm"].values[0])
            metadata["cx"] = im.shape[1]/2
            metadata["cy"] = im.shape[0]/2

        else:
            im = cv2.imread(im_name)
        coord, desc, response = compute_kp_descriptors_hardnet(im, using_ps=using_ps)
        coord = np.array(coord)
        coordinates.append(coord)
        descriptors.append(desc)
        md_list.append(metadata)
        im_list.append(im)
        response_list.append(response)
    return descriptors, coordinates, im_names, md_list, im_list, response_list


def compute_kp_descriptors_hardnet(img, nb_keypoints=None, using_ps=True):
    patch_size = 32
    batch_size = 32
    half_patch_size = patch_size // 2
    if using_ps:
        hard_net_model = HardNet2(pretrained=True)
    else:
        hard_net_model = HardNet(pretrained=True)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if nb_keypoints is not None:
        sift = cv2.SIFT_create(edgeThreshold=10,
                               nOctaveLayers=4,
                               contrastThreshold=0.02,
                               nfeatures=nb_keypoints)
    else:
        sift = cv2.SIFT_create(edgeThreshold=10,
                               nOctaveLayers=4,
                               contrastThreshold=0.02)
    kp_list, des = sift.detectAndCompute(img, None)

    coords = []
    patches = []
    for kp in kp_list:
        x, y = list(kp.pt)
        y, x = map(int, (x, y))
        patch = img[x - half_patch_size: x + half_patch_size, y - half_patch_size: y + half_patch_size]
        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
            coords.append(list(kp.pt))
            patches.append(np.expand_dims(patch, -1))
    patches = kornia.utils.image_list_to_tensor(patches).float()
    hardnet_descs = torch.zeros((patches.size(0), 128)).float()
    with torch.no_grad():
        for i in range(0, patches.size(0), batch_size):
            start = i
            end = start + batch_size
            if end > patches.size(0):
                end = patches.size(0)
            batch = patches[start: end]
            res = hard_net_model.forward(batch).detach()
            hardnet_descs[start: end] = res
    assert len(coords) == hardnet_descs.size(0)
    des = [hardnet_descs[idx].numpy() for idx in range(patches.size(0))]
    response_list = [kp.response for kp in kp_list]
    return coords, des, response_list


def build_descriptors_2d(images, images_folder="sfm_models/images", using_ps=True):
    point3did2descs = {}
    patch_size = 32
    batch_size = 32
    half_patch_size = patch_size//2
    if using_ps:
        file_name = "sfm_data_hard_net_ps.pkl"
        hard_net_model = HardNet2(pretrained=True)
    else:
        file_name = "sfm_data_hard_net.pkl"
        hard_net_model = HardNet(pretrained=True)
    my_file = Path(f"{images_folder}/{file_name}")
    if my_file.is_file():
        with open(f"{images_folder}/{file_name}", 'rb') as handle:
            p3d_id_list, p3d_desc_list, p3d_desc_list_multiple, point3did2descs = pickle.load(handle)
    else:
        for image_id in tqdm(images, desc="Loading descriptors of SfM model"):
            patches = []
            point3d_id_list = []
            image_name = images[image_id][0]
            image_name = f"{images_folder}/{image_name}"
            im = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)/255.0
            points2d_meaningful = images[image_id][1]

            for x, y, p3d_id in points2d_meaningful:
                if p3d_id > 0:
                    y, x = map(int, (x, y))
                    patch = im[x - half_patch_size: x + half_patch_size,
                            y - half_patch_size: y + half_patch_size]
                    if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                        patches.append(np.expand_dims(patch, -1))
                        point3d_id_list.append(p3d_id)
            patches = kornia.utils.image_list_to_tensor(patches).float()
            hardnet_descs = torch.zeros((patches.size(0), 128)).float()
            with torch.no_grad():
                for i in range(0, patches.size(0), batch_size):
                    start = i
                    end = start + batch_size
                    if end > patches.size(0):
                        end = patches.size(0)
                    batch = patches[start: end]
                    res = hard_net_model.forward(batch).detach()
                    hardnet_descs[start: end] = res
            assert len(point3d_id_list) == hardnet_descs.size(0)
            for idx in range(len(point3d_id_list)):
                p3d_id = point3d_id_list[idx]
                if p3d_id not in point3did2descs:
                    point3did2descs[p3d_id] = [[image_id, hardnet_descs[idx].numpy()]]
                else:
                    point3did2descs[p3d_id].append([image_id, hardnet_descs[idx].numpy()])

        p3d_id_list = []
        p3d_desc_list = []
        p3d_desc_list_multiple = []
        mean_diff = []
        for p3d_id in point3did2descs:
            p3d_id_list.append(p3d_id)
            desc_list = [du[1] for du in point3did2descs[p3d_id]]
            p3d_desc_list_multiple.append(desc_list)
            desc = np.mean(desc_list, axis=0)
            if len(desc_list) > 1:
                mean_diff.extend([np.sqrt(np.sum(np.square(desc-du))) for du in desc_list])
            p3d_desc_list.append(desc)
        print(f"Mean var. = {np.mean(mean_diff)}")
        with open(f"{images_folder}/{file_name}", 'wb') as handle:
            pickle.dump([p3d_id_list, p3d_desc_list, p3d_desc_list_multiple, point3did2descs],
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved SFM model to {images_folder}")
    return p3d_id_list, p3d_desc_list, p3d_desc_list_multiple, point3did2descs


if __name__ == '__main__':
    from colmap_io import read_points3D_coordinates, read_images

    query_images_folder = "Test line small"
    sfm_images_dir = "sfm_ws_hblab/images.txt"
    sfm_point_cloud_dir = "sfm_ws_hblab/points3D.txt"
    sfm_images_folder = "sfm_ws_hblab/images"
    image2pose = read_images(sfm_images_dir)
    # build_descriptors_2d(image2pose, sfm_images_folder, using_ps=True)
    compute_kp_descriptors_hardnet(cv2.imread("/home/sontung/work/ar-vloc/sfm_ws_hblab/images/IMG_0682.jpg"))
