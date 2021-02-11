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
	k = np.flip(np.flip(ker,axis=0),axis=1)
	out = np.full(data.shape,0)
	for i in range(1,data.shape[0]-1):
		for j in range(1,data.shape[1]-1):
			d = data[i-1:i+2,j-1:j+2];
			out[i,j] = np.sum(np.multiply(d,k))
	
	return out


def HOG(data):
	P1 = data[:data.shape[0]//2,:data.shape[1]//2]
	P2 = data[:data.shape[0]//2,data.shape[1]//2:]
	P3 = data[data.shape[0]//2:,:data.shape[1]//2]
	P4 = data[data.shape[0]//2:,data.shape[1]//2:]
	fVector = np.array([]) #feature vector
	kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
	ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	for P in [P1,P2,P3,P4]:
		hist = {0:0 , 45:0,90:0,135:0,180:0,-45:0,-90:0,-135:0}
		angMap = {0:0 , 45:45,90:90,135:135,180:180,-45:-45,-90:-90,-135:-135,-180:180,225:-135,-225:135}
		gradX = convolve2d(P,kx)
		gradY = convolve2d(P,ky)
		for i in range(P.shape[0]):
			for j in range(P.shape[1]):
				m = np.sqrt(gradX[i,j]**2 + gradY[i,j]**2)
				a = math.atan2(gradY[i,j],gradX[i,j])*180/np.pi
				if (a >= 0):
					l1 = math.floor(a/45)*45
					l2 = l1 +45

				else :
					l1 = math.ceil(a/45)*45
					l2 = l1-45
				l2 = angMap[l2]
				d1 = np.absolute(a - l1)
				d2 = np.absolute(a - l2)
				print(m,a,l1,l2,i,j,gradX[i,j],gradY[i,j])
				hist[l1] += m*(d2/(d1+d2))
				hist[l2] += m*(d1/(d1+d2))
		fVector = np.concatenate((fVector,np.array(list(hist.values()))))
	return fVector


if __name__ == '__main__':

	# kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
	# ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

	img = Image.open('straw.png')
	data = np.asarray(img)

	# H1 = convolve2d(data,kx)
	# H2 = convolve2d(data,ky)
	# plt.imshow(np.hypot(H1,H2),cmap=cm.gray_r)
	# plt.show()

	fVector = HOG(data)