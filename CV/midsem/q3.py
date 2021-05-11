from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def Saliency(r):
    sal = np.full(r.shape[:2],0)
    factor = r.shape[0]*r.shape[1]
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            sal[i,j] = np.linalg.norm(r - r[i,j],ord = np.inf,axis = 2).sum()
            print(i,j)
    return sal/factor

if __name__ == '__main__':
	rgb_img = Image.open('iiitd2.png')
	rgb_data = np.asarray(rgb_img)
	rgb_sal = Saliency(rgb_data)
	np.savetxt('S_map.csv',rgb_sal,delimiter=',')
