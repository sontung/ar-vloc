import os
import random
import torch
import pickle
from tqdm import tqdm
from PIL import Image
import pyheif
import exifread
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
from scipy.spatial import KDTree
from colmap_db_read import extract_colmap_sift

MATCHING_BENCHMARK = True

im_names = os.listdir("/home/sontung/work/sfm_ws_hblab/images")
file_ = open("debug/test.txt", "w")
for name in im_names:
    print(name, file=file_)
file_.close()