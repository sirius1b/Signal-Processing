from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import MeanShift
import time


def getMeanShift(image1,bandwidth):
	arr = np.zeros([image1.shape[0],image1.shape[1], 5])
	centeri = 0 
	centerj = 0	
	cpyx = np.zeros(image1.shape[:2])
	cpyy = np.zeros(image1.shape[:2])
	for i in range(image1.shape[0]):
		for j in range(image1.shape[1]):
			arr[i,j,:3] = image1[i,j]
			cpyx  = i - centeri
			cpyy  = j - centerj

	arr[:,:,3] = np.sqrt(cpyx**2 + cpyy**2)
	arr[:,:,4] = np.sqrt(np.arctan2(cpyy,cpyx))

	arr1 = arr.reshape([-1 , 5])
	clustering = MeanShift(bandwidth=bandwidth).fit(arr1)
	a  = clustering.labels_.reshape([image1.shape[0],image1.shape[1]])
	return a

if __name__ == '__main__':
	t1 = time.time()
	img = Image.open('1.jpg')
	image1 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HSV)
	image1 = cv2.resize(image1, (256,256))

	b1 = 0.1; b2 = 1; b3 = 3; b4 = 5

	print(1)
	a1 = getMeanShift(image1,b1)
	print(2)
	a2 = getMeanShift(image1,b2)
	print(3)
	a3 = getMeanShift(image1,  b3)
	print(4)
	a4 = getMeanShift(image1,  b4)
	print(5)
	plt.subplot(2,2,1)
	plt.imshow(a1,cmap = plt.cm.cividis)
	plt.title("bandwidth: %f"%(b1))
	plt.axis('off')

	plt.subplot(2,2,2)
	plt.imshow(a2,cmap = plt.cm.cividis)
	plt.title("bandwidth: %f"%(b2))
	plt.axis('off')
	
	plt.subplot(2,2,3)
	plt.imshow(a3,cmap = plt.cm.cividis)
	plt.title("bandwidth: %f"%(b3))
	plt.axis('off')

	plt.subplot(2,2,4)
	plt.imshow(a4, cmap = plt.cm.cividis )
	plt.title("bandwidth: %f"%(b4))
	plt.axis('off')
	plt.savefig('masks.png',dpi = 300)
	print(time.time() - t1)

	plt.show()