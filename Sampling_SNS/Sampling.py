import matplotlib.pyplot as plt
import numpy as np
from wavio import *
from math import *
import sounddevice as sd
import time



def DFT1(x): # inverse DTFT
	size= int(x.shape[0]/4)
	n = np.arange(size)
	n = n.reshape(1,size)
	arr = np.array([],dtype=np.float)
	for i in  range(int(x.shape[0])):
		W = np.exp(2j*np.pi*n*i/size)
		W1 = np.exp(2j*np.pi*i/(4*size))
		W2 = np.exp(2j*np.pi*i*2/(4*size))
		W3 = np.exp(2j*np.pi*i*3/(4*size))
		x0 = x[::4]
		x1 = x[1::4]
		x2 = x[2::4]
		x3 = x[3::4]
		arr = np.append(arr,np.dot(W,x0)+W1*np.dot(W,x1)+W2*np.dot(W,x2)+W3*np.dot(W,x3))
	return arr/(4*size)

def DFT2(x): # DTFT
	size= int(x.shape[0]/4)
	n = np.arange(size)
	n = n.reshape(1,size)
	arr = np.array([],dtype=np.complex128)
	for i in  range(0,int(x.shape[0])):
		W = np.exp(-2j*np.pi*n*i/size)
		W1 = np.exp(-2j*np.pi*i/(4*size))
		W2 = np.exp(-2j*np.pi*i*2/(4*size))
		W3 = np.exp(-2j*np.pi*i*3/(4*size))
		x0 = x[::4]
		x1 = x[1::4]
		x2 = x[2::4]
		x3 = x[3::4]
		arr = np.append(arr,np.dot(W,x0)+W1*np.dot(W,x1)+W2*np.dot(W,x2)+W3*np.dot(W,x3))
	return arr




def DFT11(x):
	size= int(x.shape[0]/4)
	n = np.arange(size)
	n = n.reshape(1,size)
	arr = np.array([],dtype=np.float)
	for i in  range(-int(x.shape[0]/2),int(x.shape[0]/2)):
		W = np.exp(2j*np.pi*n*i/size)
		W1 = np.exp(2j*np.pi*i/(4*size))
		W2 = np.exp(2j*np.pi*i*2/(4*size))
		W3 = np.exp(2j*np.pi*i*3/(4*size))
		x0 = x[::4]
		x1 = x[1::4]
		x2 = x[2::4]
		x3 = x[3::4]
		arr = np.append(arr,np.dot(W,x0)+W1*np.dot(W,x1)+W2*np.dot(W,x2)+W3*np.dot(W,x3))
	return arr/(4*size)


def Bandwidth(x):
	Max = int(max(x))
	val =  1
	for i in range(len(x)):
		if (x[i]>= Max/30):
			val = i
			break
	return [val,len(x)-val]







#############################################################

if __name__=='__main__':
	wavFile = read("Sampled.wav")
	data = wavFile.data
	rate = wavFile.rate
	size = data.shape[0]
	print("Rate-->" + str(rate))
	axis = np.arange(size)

	#plt.plot(data)
	sd.play(np.absolute(data),rate)
	time.sleep(4)
	# fTransform = np.fft.fft(data)
	fTransform = DFT2(data)
	fdata = np.absolute(fTransform)
	fangle = np.angle(fTransform)

	plt.subplot(3,2,1)
	plt.plot(axis,data)
	plt.title("Time Domain")

	plt.subplot(3,2,2)
	plt.plot(axis,fdata)
	plt.title("Magnitude Response")



	plt.subplot(3,2,3)
	plt.plot(axis,fangle)
	plt.title("Phase Response")

	plt.subplot(3,2,4)
	invSignal = DFT11(fTransform )
	plt.plot(axis,invSignal)

	plt.title("After Dtft")
	write("one.wav",invSignal,rate,sampwidth=3)
	sd.play(np.absolute(invSignal),rate)
	time.sleep(4)

	band = Bandwidth(fTransform)
	bandwidth = max(band) -min(band)

	print("Bandwidth -- >" + str(bandwidth))
	size = fTransform.shape[0]
	nSignal = np.zeros(size)#new Signal 
	minI =int(int(size)/2 - int(bandwidth)*0.8)
	maxI =int(int(size)/2 + int(bandwidth)*0.8)
	nSignal[minI:maxI ] = fTransform[minI:maxI]
	nSignalTime = DFT1(nSignal)


	plt.subplot(3,2,5)
	plt.plot(nSignalTime)


	plt.title("Time domain of 0.80 Bandwidth")
	write("two.wav",nSignalTime,rate,sampwidth=3)
	sd.play(np.absolute(nSignalTime),rate)
	time.sleep(4)

	plt.subplot(3,2,6)

	Timelimited = DFT11(fdata)
	write("third.wav",Timelimited,rate,sampwidth=3)
	plt.plot(Timelimited)
	plt.title("Zero Phase Time Domain")



	plt.show()


