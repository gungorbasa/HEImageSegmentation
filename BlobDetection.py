from matplotlib import pyplot as plt
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray
import skimage.io
import numpy as np

class BlobDetection():
    def __init__(self, name, method):
        self.method = method
        self.original_image = skimage.io.imread(name)

    def detect_blobs(self, min_sigma, max_sigma, num_sigma, threshold):
        blobs = []
        image_gray = rgb2gray(self.original_image)
        r, c = np.shape(image_gray)
        image_gray = image_gray[5:r-5,5:c-5]
        image_gray[1,1] = 1
        if self.method == 'log':
            blobs = blob_log(image_gray, min_sigma=min_sigma,
                             max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
            a = len(blobs[:])
            if a != 0:
                blobs[:, 2] = blobs[:, 2] * sqrt(2)

        elif self.method == 'dog':
            blobs = blobs_dog = blob_dog(image_gray, max_sigma=max_sigma, threshold=threshold)
            a = len(blobs[:])
            if a != 0:
                blobs[:, 2] = blobs[:, 2] * sqrt(2)
        else:
            blobs = blob_doh(image_gray, max_sigma=max_sigma, threshold=threshold)

        return blobs

    def show_blobs(self, blobs):
        fig,axes = plt.subplots(1, 1, sharex=True, sharey=True,
                                subplot_kw={'adjustable':'box-forced'})
        #axes = axes.ravel()
        #ax = axes[0]
        #axes = axes[1:]
        #ax.set_title('Detected Blobs with ' + self.method)
        #ax.imshow(self.original_image, interpolation='nearest')
        image = self.original_image

        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
            image.add_patch(c)

        viewer = ImageViewer(image)
        viewer.show()[0][0]
