from PIL import Image

#Method 1
# image = Image.open('Project1.png')
# print(image.format,image.size, image.mode)
# # image.show()

#Method 2
from numpy import asarray
import numpy as np
# data = asarray(image)
# print(type(data),data.shape)


# image2 = Image.fromarray(data)
# print(type(image2))
# print(image2.mode)
# print(image2.size)	
# image2.show()



# to determine rgb indexes
# data = np.full((100,100,3),[0,255,0],dtype='uint8')
# img2 = Image.fromarray(data)
# img2.show()