from PIL import Image
import os
import pyheif

directory = "/home/sontung/work/ar-vloc/Test line-20211207T083302Z-001/Test line"
files = os.listdir(directory)
for name in files:
    if name.split(".")[-1] == "HEIC":
        heif_file = pyheif.read(f"{directory}/{name}")
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
            )

        image.save(f"{directory}/{name.split('.')[0]}.jpg",
                   "JPEG")
    else:
        image = Image.open(f"{directory}/{name}")
        image.save(f"{directory}/{name.split('.')[0]}.jpg",
                   "JPEG")
