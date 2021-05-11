from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm


def getHist(img, bins):
	# histogram storing count of pixels belonging to the particular bin
	img = (img - img.min()) /(img.max() - img.min())
	hist = np.zeros((bins,bins,bins))
	img1 = img*(bins-1)
	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
			index = np.round(img1[i,j]).astype(int) 
			hist[index[0],index[1],index[2]] +=1
	return hist

def getSimilarity(hist1, hist2):
	# similarity score for each bin
	sim = np.zeros(hist1.shape)
	for i in range(hist1.shape[0]):
		for j in range(hist1.shape[1]):
			for k in range(hist1.shape[2]):
				sim[i,j,k] = 	min(hist1[i,j,k],hist2[i,j,k])/(max(hist1[i,j,k],hist2[i,j,k]) + 1e-10)
	return sim

def similarityMap(img,sim,bins):
	## returns: similarity map, similarity mask
	##
	img = (img - img.min()) /(img.max() - img.min())
	img2 = np.zeros((img.shape[0],img.shape[1]))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			index = np.round((bins-1)*img[i,j]).astype(int)
			img2[i,j] = sim[index[0],index[1],index[2]]

	ret, th = cv2.threshold((img2*(2**16)).astype(np.uint16), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU); 
	return img2,th

image1 = np.asarray(Image.open('b11.png'))
image2 = np.asarray(Image.open('b12.png'))
img1 = np.float32(cv2.cvtColor(np.asarray(image1), cv2.COLOR_RGBA2RGB))
img2 = np.float32(cv2.cvtColor(np.asarray(image2), cv2.COLOR_RGBA2RGB))

bins = 10
hist1 = getHist(img1, bins = bins)
hist2 = getHist(img2, bins = bins)

sim = getSimilarity(hist1,hist2)
s1,s1t = similarityMap(img1,sim,bins = bins)
s2,s2t = similarityMap(img2,sim,bins = bins)


plt.subplot(2,2,1)
plt.imshow(image1)
plt.axis('off')
plt.title("Image 1")

plt.subplot(2,2,2)
plt.imshow(s1,cmap = plt.cm.gray)
plt.axis('off')
plt.title("Similarity Map(Image 1)")


plt.subplot(2,2,3)
plt.imshow(image2)
plt.axis('off')
plt.title("Image 2")

plt.subplot(2,2,4)
plt.imshow(s2,cmap = plt.cm.gray)
plt.axis('off')
plt.title("Similarity Map(Image 2)")

plt.show()
