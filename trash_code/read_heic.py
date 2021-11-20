from PIL import Image
import os
import pyheif

files = os.listdir("/home/sontung/work/hblab-office-20211120T030010Z-001/hblab-office")
for name in files:
    if name.split(".")[-1] == "HEIC":
        heif_file = pyheif.read(f"/home/sontung/work/hblab-office-20211120T030010Z-001/hblab-office/{name}")
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
            )

        image.save(f"/home/sontung/work/hblab-office-20211120T030010Z-001/hblab-office2/{name.split('.')[0]}.jpg",
                   "JPEG")
    else:
        image = Image.open(f"/home/sontung/work/hblab-office-20211120T030010Z-001/hblab-office/{name}")
        image.save(f"/home/sontung/work/hblab-office-20211120T030010Z-001/hblab-office2/{name.split('.')[0]}.jpg",
                   "JPEG")
