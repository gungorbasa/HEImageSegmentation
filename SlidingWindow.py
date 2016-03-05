from skimage.transform import pyramid_gaussian
import scipy.misc as ms
import cv2

# Sources:
# http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
# http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
#

class Samples():
    def __init__(self, image_path, width, height):
        self.image = cv2.imread(image_path)
        self.win_width = width
        self.win_height = height


    def sliding_window(self, image, stepSize):
        windowSize = [self.win_height, self.win_height]
        for y in xrange(0, image.shape[0], stepSize):
            for x in xrange(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    def pyramid(self, image, scale, smallest_height, smallest_width):
        for (i, resized) in enumerate(pyramid_gaussian(image, downscale=scale)):
            # if the image is too small, break from the loop
            if resized.shape[0] < smallest_height or resized.shape[1] < smallest_width:
                break
            yield(resized)

    # save_path = /Labels/HE_label/ or /Labels/Nuclei_label
    def pyramid_sliding_window(self, save_path, smallest_image_h, smallest_image_w):
        image = self.image
        winH = self.win_height
        winW = self.win_width

        i = 0
        for resized in self.pyramid(image, 1.5, smallest_image_h, smallest_image_w):
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in self.sliding_window(resized, stepSize=16):
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                print(save_path + str(i) + '.png')
                ms.imsave(save_path + str(i) + '.png', window)
                i += 1
