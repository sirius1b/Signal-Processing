from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

img = Image.open('Cap1.png')
image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2RGB)

numSegments = 50

segments = slic(image, n_segments = numSegments, sigma = 10,start_label= 1)
K = len(np.unique(segments))

centers = np.zeros((K,3))

for i in range(K):
    si = segments == i
    si = np.repeat(np.expand_dims(si,axis= 2), 3, axis = 2)
    data = np.ma.array(image, mask = ~si)
    centers[i] = data.mean(axis = (0,1))


# print(segments)
# input()

# fig = plt.figure("Superpixels -- %d segments" % (numSegments))
# ax = fig.add_subplot(1, 1, 1)
# ax.imshow(mark_boundaries(image, segments))
# plt.axis("off")

# print(np.unique(segments))
# plt.show()