from BlobDetection import BlobDetection
import os
from os import path
import numpy as np

def grab_files(directory):
    for name in os.listdir(directory):
        full_path = os.path.join(directory, name)
        if os.path.isdir(full_path):
            for entry in grab_files(full_path):
                yield entry
        elif os.path.isfile(full_path):
            yield full_path

path = 'Labels/Train/Nuclei_label'
files = grab_files(path)
num_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
# num_files -= 1
print(num_files)
y = np.zeros(num_files)
j = 0
for f in files:
    if ".DS_Store" in f:
        continue
    print f
    b = BlobDetection(f, 'log')
    blobs = b.detect_blobs(min_sigma=10, max_sigma=50, num_sigma=10, threshold=.15)
    num_blobs = len(blobs)
    if num_blobs > 0:
        y[j] = 1
    j += 1

np.savetxt("./Labels/train_labels.csv", y, delimiter=",")
