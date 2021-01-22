from PIL import Image
import numpy as np

def visit(data,discovered,q,imx,imn,jmx,jmn):
	# q = [(i,j)]
	while (len(q) != 0):
		p = q.pop(); i = p[0]; j = p[1]
		for a in [-1,0,1]:
			for b in [-1,0,1]:
				if (discovered[i+a,j+b] == False and data[i+a,j+b] == True and i+a>=imn and i+a<=imx and j+b >=jmn and j+b <= jmx ):
					discovered[i+a,j+b] = True
					q.insert(0,(i+a,j+b))
	# return discovered



def boolConnComp(data):
	i= 0 ; j = 0 ; i_max = data.shape[0]-1 ; j_max = data.shape[1]-1
	discovered = np.full(data.shape,False,dtype=bool)
	count = 0
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if (data[i,j] == True and discovered[i,j] == False):
				count += 1
				visit(data,discovered,[(i,j)],data.shape[0]-1,0,data.shape[1]-1,0) # shallow memory
	return count

if __name__ == '__main__':
	img = Image.open('Project1.png')
	data = np.asarray(img)
	print("Number of objects in image: "+str(boolConnComp(data)))



