from SlidingWindow import Samples

width = 30
height = 30

he = Samples('./HE_label.tif', height, width)
he.pyramid_sliding_window('./Labels/HE_label/', 1000, 1000)

nuc = Samples('./Nuclei_label.tif', height, width)
nuc.pyramid_sliding_window('./Labels/Nuclei_label/', 1000, 1000)
