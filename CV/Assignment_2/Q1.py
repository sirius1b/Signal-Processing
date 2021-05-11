"""
Author: Lavanya Verma(2018155)
email : lavanya18155@iiitd.ac.in
Solution of Q1, CV Assignment 2. 
CV @ IIITD Winter 2021
"""


# run pip install fuzzy-c-means / pip3 install fuzzy-c-means
import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt
from PIL import Image
import time

def RGBXY(image):
    shape = list(image.shape); shape[2] = 5
    img = np.zeros(shape)
    img[:,:, :3] = image/255
    indexes = np.array([[i,j] for i in range(shape[0]) for j in range(shape[1])])
    img[:,:,3:] = indexes.reshape((shape[0],shape[1],2))
    img[:,:,3] = img[:,:,3] / shape[0]
    img[:,:,4] = img[:,:,4] / shape[1]
    return img.reshape([-1, 5]), indexes

c = int(input("Numbers of clusters(c): "))
img = np.asarray(Image.open('1558014721_E7jyWs_iiit_d.jpg')).copy()
t1 = time.time()
features,indexes = RGBXY(img)
fcm = FCM(n_clusters=c)
fcm.fit(features)
fcm_centers = fcm.centers
fcm_labels = fcm.predict(features)
print(fcm_centers,fcm_centers.shape)
img = fcm_centers[fcm_labels,:3].reshape((img.shape[0],img.shape[1],3))
print("Time taken: %f"%( time.time() - t1))
plt.imshow(img)
plt.axis('off')
plt.show()

