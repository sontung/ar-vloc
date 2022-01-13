from os import listdir
from os.path import isfile, join
import sys

mypath = "../sfm_ws_hblab/images"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
sys.stdout = open("train_images.txt", "w")
for img in onlyfiles:
    print(img)