import os
import random
from Helper import Helper
import numpy as np
import shutil


path = "/scratch/basag/"
imgs = os.listdir(path + "original/HE_label")
num_imgs = len(imgs)
y = Helper.load_labels(path + "labels.csv")
test_y = np.zeros(shape=(50000,1))
l = []

for i in xrange(50000):
    r = random.randint(0, num_imgs)
    while r in l:
        r = random.randint(0, num_imgs)
    l.append(r)
    f = str(r)
    f = f.zfill(8)
    shutil.move(path + "original/HE_label/" + f + ".png", path + "test_data/" + f + ".png")
    test_y[i] = y[r]
    np.delete(y, r)

np.savetxt("/scratch/basag/train_labels.csv", y, delimiter=",")
np.savetxt("/scratch/basag/test_labels.csv", test_y, delimiter=",")
