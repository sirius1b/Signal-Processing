#!/usr/binda/python3
# Lavanya Verma(2018155)
from PIL import Image, ImageOps
import numpy as np
import math
import matplotlib.pyplot as plt
from  matplotlib import cm

def convolve2d(data,ker):
	""" data: Main image matrix
		ker : Kernel
	"""
	k = np.flip(np.flip(ker,axis=0),axis=0)
	out = np.full(data.shape,0)
	for i in range(1,data.shape[0]-1):
		for j in range(1,data.shape[1]-1):
			d = data[i-1:i+2,j-1:j+2];
			out[i,j] = np.sum(np.multiply(d,k))
	return out


def otsu(data):
	"""
	data: grayscale data of image
	returns bitmask 
	"""
	SS = []
	for t in range(int(np.floor(data.min())),int(np.ceil(data.max()))+1):
		smaller = np.ma.masked_greater_equal(data,t)
		greater = np.ma.masked_less(data,t)
		if (smaller.count() == 0 or greater.count() == 0):
			continue
		Score = smaller.count()*smaller.var() + greater.count()*greater.var()
		SS.insert(0,[t,Score])
	SS = sorted(SS,key = lambda x : x[1])
	threshold = SS[0][0]
	print(threshold)
	mask = np.ma.masked_greater_equal(data,threshold).mask
	return mask	

if __name__ == '__main__':

	kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
	ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

	# img = Image.open('cow.jpeg')
	# img_g = ImageOps.grayscale(img)
	# data = np.asarray(img_g)


	data = np.array([[1 ,1 ,9,9,9],[ 1,1,9,9,9],[1,1,9,9,9],[1,1,1,1,1],[1,1,1,1,1]])
	print(data)

	H1 = convolve2d(data,kx)
	H2 = convolve2d(data,ky)
	print(H1)
	print(H2)
	# Hxx = np.multiply(H1,H1)
	# Hyy = np.multiply(H2,H2)
	# Hxy = np.multiply(H1,H2)

	# plt.imshow(np.hypot(H1,H2),cmap=cm.gray_r)
	# plt.show()

	M = np.array([[0,0],[0,0]])
	for i in range(H1.shape[0]):
		for j in range(H2.shape[1]):
			M[0,0] += math.pow(H1[i,j],2)
			M[0,1] += H1[i,j]*H2[i,j]
			M[1,0] += H1[i,j]*H2[i,j]
			M[1,1] += math.pow(H2[i,j],2)
	print(M)



	# d = np.full(data.shape,0)
	# k = 0.06
	# for i in range(data.shape[0]):
	# 	for j in range(data.shape[1]):
	# 		M = np.array([[Hxx[i,j],Hxy[i,j]],[Hxy[i,j],Hyy[i,j]]])
	# 		eig,v = np.linalg.eig(M)
	# 		R = np.linalg.det(M) - k * np.trace(M)**2
	# 		d[i,j] = R
	# # mask = otsu(d)
	# d = (d - d.min())/(d.max() - d.min())*100
	# plt.imshow(d,cmap=cm.inferno)

	# plt.show()
