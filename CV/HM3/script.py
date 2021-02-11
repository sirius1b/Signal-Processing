from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def saliency_map(dat): #naive implementation
	S_map = np.full(dat.shape[0:2],-1)
	for i in range(dat.shape[0]):
		for j in range(dat.shape[1]):
			print(i,j)
			S_ij= 0
			for n in range(dat.shape[0]):
				for m in range(dat.shape[1]):
					d = np.linalg.norm([i-n,j-m])
					Ii_j = np.linalg.norm(dat[i][j] - dat[n][m])*np.exp(-d)
					S_ij += Ii_j
			S_map[i,j] = S_ij
	return S_map

def saliency_map_1(dat): #somewhat optimized computations
	S_map = np.full(dat.shape[0:2],0)
	for i in range(dat.shape[0]):
		for j in range(dat.shape[1]):
			print(i,j)
			for n in range (i,-1,-1):
				for m in range(j,-1,-1):
					d = np.linalg.norm([i-n,j-m])
					Ii_j = np.linalg.norm(dat[i][j] - dat[n][m])*np.exp(-d)
					S_map[n,m] += Ii_j
					S_map[i,j] += Ii_j
	return S_map

if __name__== '__main__':
	img = Image.open('download.jpg')
	dat = np.asarray(img)
	# S_map = saliency_map_1(dat)
	# np.savetxt('S_map.csv',S_map,delimiter = ',')
	S_map = np.genfromtxt('S_map.csv',delimiter=',')
	mx = S_map.max() ;  mn  = S_map.min()
	# s = np.array(list(map(lambda x: np.ceil((x- mn)*255/mx),S_map)))
	# sal_img = Image.fromarray(s)
	# sal_img.show()

	plt.imshow(S_map)
	plt.show()