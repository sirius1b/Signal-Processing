#!/usr/bin/python3
# Lavanya Verma(2018155)
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
from  matplotlib import cm

def convolve2d(data,ker):
	""" data: Main image matrix
		ker : Kernel
	"""
	# k = np.flip(np.flip(ker,axis=0),axis=1)
	out = np.full(data.shape,0)
	for i in range(1,data.shape[0]-1):
		for j in range(1,data.shape[1]-1):
			d = data[i-1:i+2,j-1:j+2];
			out[i,j] = np.sum(np.multiply(d,ker))
	
	return out

# if __name__ == '__main__':

# 	# kx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
# 	# ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

# 	# img = Image.open('straw.png')
# 	# data = np.asarray(img)

# 	# H1 = convolve2d(data,kx)
# 	# H2 = convolve2d(data,ky)
# 	# plt.imshow(np.hypot(H1,H2),cmap=cm.gray)
# 	# plt.show()

# 	