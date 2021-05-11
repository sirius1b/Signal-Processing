"""
Author: Lavanya Verma(2018155)
email : lavanya18155@iiitd.ac.in
Solution of Q2, CV Assignment 2. 
CV @ IIITD Winter 2021
"""

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.segmentation import slic

def contrastcue(image,K_means,numSegments = 100):
    mask = slic(image, n_segments = numSegments, sigma = 10,start_label= 1)
    K = len(np.unique(mask))
    centers = np.zeros((K,3),dtype= np.float32 )
    for i in range(1,K+1):
        si = mask == i
        si = np.repeat(np.expand_dims(si,axis= 2), 3, axis = 2)
        data = np.ma.array(image, mask = ~si)
        centers[i-1] = data.mean(axis = (0,1))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,Kcenter=cv2.kmeans(centers,K_means,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    bins = np.zeros((K_means,))
    counts  = np.array([np.sum(label == i) for i in range(K_means)])
    sal = mask.copy()
    mask1 = mask.copy()
    for k in range(K_means):  
        d = Kcenter - Kcenter[k]
        bins[k] = sum(counts*np.sqrt(np.sum(d*d,1)))/sum(counts)
        indexs = np.where(label == k)
        superPixels = centers[indexs]
        for i in range(len(superPixels)):
            slicK = np.where(centers == superPixels[i])[0]
            sal[mask == slicK ] = bins[k]
            mask1[mask == slicK] = k
    # sal = (sal - sal.min())/(sal.max() - sal.min())
    return sal ,mask1,K_means

def spatialcue(image1,K,mask):
    Z = np.full((image1.shape[0],image1.shape[1],2),[0,0],dtype=np.float32)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = [i/Z.shape[0],j/Z.shape[1]]
    z1 = Z - Z[int(Z.shape[0]/2),int(Z.shape[1]/2)]
    sigma = 0.28
    N = np.exp(-np.sum(z1*z1,2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    bins = np.array([np.sum(N*(mask == i))/np.sum(mask == i) for i in range(K)])
    sal2 = np.zeros(mask.shape)
    for i in range(K):
        sal2[mask == i] = bins[i]
    # sal2 = (sal2 - sal2.min())/(sal2.max() - sal2.min())
    return sal2

if __name__ == '__main__':
    image1 = np.asarray(Image.open('1558014721_E7jyWs_iiit_d.jpg'))
    n_segments = 100
    K = 8
    sal1,mask,K = contrastcue(image1,K,n_segments)
    sal2 = spatialcue(image1,K,mask)
    plt.subplot(2,2,1)
    plt.imshow(sal1,cmap = cm.gray)
    plt.title("Contrast Cue")
    plt.axis('off')
    plt.subplot(2,2,2)
    plt.imshow(sal2,cmap = cm.gray)
    plt.title("Spatial Cue")
    plt.axis('off')
    plt.subplot(2,2,3)
    plt.imshow(sal1*sal2,cmap = cm.gray)
    plt.title("(Contrast Cue) .* (Spatial Cue)")
    plt.axis('off')
    plt.subplot(2,2,4)
    plt.imshow(image1)
    plt.title("Image")
    plt.axis('off')
    plt.show()

