#Lavanya Verma(2018155)
# HM 13
# Saliency Computation from Contrast & Spatial Cues
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

def contrastcue(image1,K):
	image = image1.reshape([-1,3])
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(image,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint16(center)
	res = center[label.flatten()]
	res2 = res.reshape(image1.shape)
	mask = label.reshape(image1.shape[:2])
	bins = np.zeros((K,))
	counts  = np.array([np.sum(mask == i) for i in range(K)])
	sal = mask.copy()
	for k in range(K):
		d = center - center[k]
		bins[k] = sum(counts*np.sqrt(np.sum(d*d,1)))/sum(counts)
		sal[mask == k ] = bins[k]
	return sal,mask


def spatialcue(image1,K,mask):
	Z = np.full((image1.shape[0],image1.shape[1],2),[0,0],dtype=np.float32)
	for i in range(Z.shape[0]):
		for j in range(Z.shape[1]):
			Z[i,j] = [i/Z.shape[0],j/Z.shape[1]]
	z1 = Z - Z[int(Z.shape[0]/2),int(Z.shape[1]/2)]
	sigma = 0.1
	N = np.exp(-np.sum(z1*z1,2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
	bins = np.array([np.sum(N*(mask == i))/np.sum(mask == i) for i in range(K)])
	sal2 = np.zeros(mask.shape)
	for i in range(K):
		sal2[mask == i] = bins[i]
	return sal2

if __name__ == '__main__':
	img = Image.open('1558014721_E7jyWs_iiit_d.jpg')
	image1 = np.float32(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2RGB))
	K = 8
	sal1,mask = contrastcue(image1,K)
	sal2 = spatialcue(image1,K,mask)
	plt.subplot(1,3,1)
	plt.imshow(sal1,cmap = cm.gray)
	plt.title("Contrast Cue")
	plt.subplot(1,3,2)
	plt.imshow(sal2,cmap = cm.gray)
	plt.title("Spatial Cue")
	plt.subplot(1,3,3)	
	plt.imshow(sal1*sal2,cmap = cm.gray)
	plt.title("Saliency")
	plt.show()

