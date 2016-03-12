from SlidingWindow import Samples
from skimage.transform import pyramid_gaussian
import scipy.misc as ms
import cv2

width = 64
height = 64

nuclei_label = cv2.imread('./Nuclei_label.tif')
nuclei_test = nuclei_label[0:2000, :]
nuclei_train = nuclei_label[2000:, :]
cv2.imwrite("nuclei_test.png", nuclei_test)
cv2.imwrite("nuclei_train.png", nuclei_train)

he_label = cv2.imread('./HE_label.tif')
he_test = he_label[0:2000, :]
he_train = he_label[2000:, :]
cv2.imwrite("he_test.png", he_test)
cv2.imwrite("he_train.png", he_train)


nuc_train = Samples('./nuclei_train.png', height, width)
nuc_train.pyramid_sliding_window('./Labels/Train/Nuclei_label/', 1000, 1000)
#
#
he_train = Samples('./he_train.png', height, width)
he_train.pyramid_sliding_window('./Labels/Train/HE_label/', 1000, 1000)

nuc_train = Samples('./nuclei_test.png', height, width)
nuc_train.pyramid_sliding_window('./Labels/Test/Nuclei_label/', 1000, 1000)
#
#
he_train = Samples('./he_test.png', height, width)
he_train.pyramid_sliding_window('./Labels/Test/HE_label/', 1000, 1000)
