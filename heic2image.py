import numpy as np
import os
import pyheif
import skimage.io
from PIL import Image


def load_2d_queries_generic(folder="Test line"):
    im_names = os.listdir(folder)
    new_dir = "/home/sontung/work/recon_models/office/images"
    for idx, name in enumerate(im_names):
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
            skimage.io.imsave(f"{new_dir}/{im_name.split('/')[-1]}.jpg", im)


if __name__ == '__main__':
    load_2d_queries_generic()
