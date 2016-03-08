import os
import random
from Helper import Helper
import numpy as np
import shutil


path = "/scratch/basag/"
imgs = os.listdir(path + "HE_labels")
num_imgs = len(imgs)
y = Helper.load_labels(path + "labels.csv")

l = []

for i in xrange(50000):
    r = random.randint(0, num_imgs)
    while r in l:
        r = random.randint(0, num_imgs)

    f = str(r)
    f = f.zfill(8)
    shutil.move(path + "HE_labels/" + f + ".png", path + "test_data/" + f + ".png")
