#!/usr/binda/python3
# Lavanya Verma(2018155)
from PIL import Image, ImageOps
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib import cm


def hist1(img_patch):
	table = np.zeros(256)
	for i in range(0,256):
		table[i] = np.count_nonzero(img_patch == i)
	table = table/table.max()
	return table


if __name__ == '__main__':
	img = Image.open('dog2.png')
	data = np.asarray(img)
	d1 = data.copy()
	fg_patch_px = [(150,125),(250,175)]
	bg_patch_px = [(300,40),(400,100)]
	cv.rectangle(data,fg_patch_px[0],fg_patch_px[1],(0,255,0),2)
	cv.rectangle(data,bg_patch_px[0],bg_patch_px[1],(0,0,0),2)
	fg_patch = d1[fg_patch_px[0][1]:fg_patch_px[1][1],fg_patch_px[0][0]:fg_patch_px[1][0]]
	bg_patch = d1[bg_patch_px[0][1]:bg_patch_px[1][1],bg_patch_px[0][0]:bg_patch_px[1][0]]
	fg_table = hist1(fg_patch)
	bg_table = hist1(bg_patch)


	d2 = d1.copy() ; d3 = d1.copy(); d4 = d1.copy()
	# background: 0 , foreground : 255
	for i in range(256):
		d2 [d1 == i] = 255*fg_table[i]
		d3 [d1 == i] = 255*(1 - bg_table[i])
		val = 0 if (fg_table[i] + 1 - bg_table[i])/2 <0.5 else 255
		d4 [d1 ==i] = val


	plt.subplots_adjust(top=0.97,
		bottom=0.035,
		left=0.035,
		right=0.95,
		hspace=0.185,
		wspace=0.025)


	plt.subplot(2,2,1)
	plt.imshow(data,cmap =cm.gray)
	plt.subplot(2,2,2)
	plt.imshow(d2,cmap = cm.gray)
	plt.subplot(2,2,3)
	plt.imshow(d3,cmap = cm.gray)
	plt.subplot(2,2,4)
	plt.imshow(d4,cmap = cm.gray)
	# plt.savefig('fg_bg_segement.png', dpi=400, bbox_inches='tight')
	plt.show()
