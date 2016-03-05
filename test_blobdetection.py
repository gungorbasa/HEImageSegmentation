from BlobDetection import BlobDetection

b = BlobDetection('Nuclei_label.tif', 'log')
print('Detecting Blobs. THis may take long..')
blobs = b.detect_blobs(max_sigma=30, num_sigma=10,threshold=.1)
b.show_blobs(blobs)

