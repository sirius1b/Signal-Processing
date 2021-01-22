from PIL import Image, ImageOps
import numpy as np

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

def classify(img,mask):
	dat = np.asarray(img)
	bl1 = np.full(dat.shape,[0,0,255],dtype='uint8'); 	
	# boundary pixel count
	bl1[:,(0,mask.shape[1]-1)] = [216,8,255]
	bl1[(0,mask.shape[0]-1),:] = [216,8,255]
	bl2 = bl1.copy()
	bl1[mask] = dat[mask]
	bl2[~mask] = dat[~mask]


	# ---- count of boundary pixels
	c1 = np.ma.masked_not_equal(bl1,[216,8,255]).count()
	c2 = np.ma.masked_not_equal(bl2,[216,8,255]).count() # gives the count of non replaced boundary pixel, represented in pink
	flag = 0
	if (c1 < c2):
		flag = 2
	elif (c2 < c1):
		flag = 1
	else :
		if (mask[mask.shape[0]//2],mask[mask.shape[1]//2]) :
			flag = 1 
		else :
			flag = 2

	if (flag == 1):
		foreground = Image.fromarray(bl1)
	else:
		foreground = Image.fromarray(bl2)

	foreground.show()

if __name__ == '__main__':
	img = Image.open('pic_main.jpeg')
	img_g = ImageOps.grayscale(img)

	data1 = np.asarray(img_g)

	mask = otsu(data1)
	classify(img,mask)
