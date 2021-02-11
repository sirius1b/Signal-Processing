# Lavanya Verma(2018155)
from PIL import Image
import numpy as np


def LBP(data):
	P1 = data[:data.shape[0]//2,:data.shape[1]//2]
	P2 = data[:data.shape[0]//2,data.shape[1]//2:]
	P3 = data[data.shape[0]//2:,:data.shape[1]//2]
	P4 = data[data.shape[0]//2:,data.shape[1]//2:]
	fVector = np.array([]) #feature vector
	d = {0:(-1,-1),1:(-1,0),2:(-1,1),3:(0,1),4:(1,1),5:(1,0),6:(1,-1),7:(0,-1)}
	for P in [P1,P2,P3,P4]:
		hist = np.full((256),0)
		for i in range(1,P.shape[0]-1):
			for j in range(1,P.shape[1]-1):
				val = 0
				for v in range(8):
					ic = i + d[v][0]; jc = j + d[v][1]
					val += (1 if P[i,j] >= P[ic,jc] else 0)*2**(v)
				hist[val] += 1
		fVector = np.concatenate((fVector,hist))
	return fVector		

if __name__ == '__main__':
	img = Image.open('straw.png')
	data = np.asarray(img)
	fVector = LBP(data)