#!/usr/binda/python3
# Lavanya Verma(2018155)
from PIL import Image, ImageOps
import numpy as np
import math
import matplotlib.pyplot as plt






if __name__ == '__main__':
	img = Image.open('dog2.png')
	data = np.asarray(img)
	plt.imshow(data)
	plt.show()
