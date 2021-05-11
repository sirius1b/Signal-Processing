import cv2 
import numpy as np

def kmeans(image1,K):
	image = image1.astype(np.float32).reshape([-1,3])
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center=cv2.kmeans(image,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape(image1.shape)
	return res2



cap = cv2.VideoCapture(0)

if not cap.isOpened():
	raise IOError("Cannot open webcam")
while True:
	ret, frame = cap.read()
	frame = cv2.resize(frame,None,fx = 0.4 , fy = 0.4)
	fr = kmeans(frame,10)
	cv2.imshow('Input',fr)

	c = cv2.waitKey(1)
	if c == 27:
		break
cap.release()
cv2.destroyAllWindows()