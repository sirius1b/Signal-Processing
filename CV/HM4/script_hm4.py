from PIL import Image
import numpy as np
import cv2 as cv


def otsu(data):
	"""
	data: grayscale data of image
	returns threshold
	"""
	mVal = np.inf ; threshold = 0
	for i in range(len(data)):
		arr1 = data[0:i+1] ; arr2 = data[i+1:]
		score = (i+1)*np.var(data[0:i+1])
		if (len(arr2) != 0):
			score += (len(data) - i - 1)*np.var(data[i+1:])
		if (mVal > score):
			mVal = score 
			threshold = i
	return threshold

def replace(q,dataColor,blue,discovered):
	""" comp: pixel[(x,y)]
		dataColor: original rgb image
		blue: dummy image on which replace dummy values with rgb of mask
	"""
	minX = np.inf; maxX = -1 ;minY = np.inf ;maxY = -1
	imn = 0 ; imx = blue.shape[0] -1 ; jmn = 0 ; jmx = blue.shape[1] - 1
	while (len(q) != 0):
		p = q.pop(); i = p[0]; j = p[1]
		minX = min(j, minX);maxX = max(j,maxX)
		minY = min(i,minY) ; maxY = max(i,maxY)
		for a in [-1,0,1]:
			for b in [-1,0,1]:
				if (discovered[i+a,j+b] == False and data[i+a,j+b] == True and i+a>=imn and i+a<=imx and j+b >=jmn and j+b <= jmx ):
					discovered[i+a,j+b] = True
					q.insert(0,(i+a,j+b))
					blue[i,j] = dataColor[i,j]
	return minX ,maxX,minY,maxY
	
def visit(dataColor,data,discovered,q,imx,imn,jmx,jmn):
	# q = [(i,j)]
	descriptors= np.array([0.0,0,0,0,0,0]) # red intensity count, green intensity count, blue intensity count, x values sum , y value sum, count
	while (len(q) != 0):
		p = q.pop(); i = p[0]; j = p[1]
		for a in [-1,0,1]:
			for b in [-1,0,1]:
				if (discovered[i+a,j+b] == False and data[i+a,j+b] == True and i+a>=imn and i+a<=imx and j+b >=jmn and j+b <= jmx ):
					discovered[i+a,j+b] = True
					q.insert(0,(i+a,j+b))
					descriptors[0] += dataColor[i,j,0] 
					descriptors[1] += dataColor[i,j,1]
					descriptors[2] += dataColor[i,j,2]
					descriptors[3] += i
					descriptors[4] += j
					descriptors[5] += 1
	descriptors[0:5] = descriptors[0:5] /descriptors[5] #averaging (mean computation)
	return descriptors

def boolConnComp(data,dataColor):
	i= 0 ; j = 0 ; i_max = data.shape[0]-1 ; j_max = data.shape[1]-1
	discovered = np.full(data.shape,False,dtype=bool)
	# count = 0
	clusters = []
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if (data[i,j] == True and discovered[i,j] == False):
				# count += 1
				arr = visit(dataColor,data,discovered,[(i,j)],data.shape[0]-1,0,data.shape[1]-1,0) # shallow memory
				arr = np.concatenate((arr,np.array([i,j])))
				clusters.append(arr)
	clusters = np.array(clusters)
	clusters = clusters[np.argsort(clusters[:,5])] # Part 1 : largest component can be accessed on the last index
	dist = np.full((clusters.shape[0],1),0.)
	for i in range(clusters.shape[0]):
		dist[i] = np.linalg.norm(clusters[i,0:5] - clusters[-1,0:5]) # Part 2: Euclidean Norm based on mean (R,G,B,X,Y),
		dist[i] = dist[i]/(clusters[i,5]*clusters[-1,5]) # Part 3: divided by no. of pixels in conn. components(P1,P2)
	dist = dist/np.max(dist) # Part 4 : normalisation (0,1)	
	threshold = otsu(dist) # Part 4:Applying otsu to compute optimal threshold
	
	# Part 4: replacing pixels estimated to be object with their respective RGB values in dummy image(background)
	blue = np.full(dataColor.shape,[0,0,0],dtype='uint8')
	dS = np.full(data.shape,False,dtype=bool)
	minX = np.inf; maxX = -1; minY  = np.inf ;maxY  = -1 # for localisation(BBOX)
	for i in range(threshold+1,len(dist)):
		M = replace([(int(clusters[i,6]),int(clusters[i,7]))],dataColor,blue,dS)
		minX = min(minX,M[0]) 
		maxX = max(maxX,M[1])
		minY = min(minY,M[2]) 
		maxY = max(maxY,M[3])

	# Part 5: Localisation of object (BBOX)
	cv.rectangle(blue,(minX,minY),(maxX,maxY),(0,128,0),3)
	img = Image.fromarray(blue)
	img.show()
	

if __name__ == '__main__':
	img = Image.open('mask.png')
	imgColor = Image.open('Blue_Winged.jpg')
	data = np.asarray(img)
	dataColor = np.asarray(imgColor)
	boolConnComp(data,dataColor)

