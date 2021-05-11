# Lavanya Verma(2018155)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def LBP_Map(data):
	data1 = np.full(data.shape,0) # Zero Padding
	d = {0:(-1,-1),1:(-1,0),2:(-1,1),3:(0,1),4:(1,1),5:(1,0),6:(1,-1),7:(0,-1)}
	eps = 1e-15
	for i in range(1,data.shape[0]-1):
		for j in range(1,data.shape[1]-1):
			for v in range(8):
				ic = i + d[v][0]; jc = j + d[v][1]
				# val += int(min(data[ic,jc],data[i,j])/(max(data[ic,jc],data[i,j]) + eps))*2**(v)
				data1[i,j] += int((data[ic,jc]+eps)/(data[i,j] + eps))*2**(v)
	return data1 

if __name__ == '__main__':
	img = Image.open('iiitd1.png')
	data = np.asarray(img)
	data1 = LBP_Map(data)
	plt.imshow(data1,cmap = cm.gray)
	plt.show() 
