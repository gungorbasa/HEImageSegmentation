from BlobDetection import BlobDetection

b = BlobDetection('Labels/Nuclei_label/00000000.png', 'log')
print('Detecting Blobs. This may take long..')
blobs = b.detect_blobs(min_sigma=10, max_sigma=50, num_sigma=10, threshold=.15)
print(len(blobs))
# b.show_blobs(blobs)
