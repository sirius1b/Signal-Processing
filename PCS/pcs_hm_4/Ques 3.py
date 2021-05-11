import matplotlib.pyplot as plt, numpy as np , math, scipy.signal as signal,copy,sys

plt.subplots_adjust(left=0.05,bottom=0.06,right=0.96,top=0.94,wspace=0.13,hspace=0.14)

def Samples_To_Bits(mp,n,x):
	dv=2*mp/(2**n)
	l=np.arange(-mp,mp,dv)+dv/2
	enc={}
	for i in range(len(l)):
		z = bin(i)[2:]
		z='0'*(int(math.log2(len(l)))-len(z))+z
		enc[l[i]]=z
	for i in range(len(x)):
		dist = np.absolute(l-x[i])
		index = np.where(dist==min(dist))[0][0]
		x[i]= l[index]

	z = copy.deepcopy(x)
	x = list(map(lambda t: enc[t],x))
	return enc,x,z

def Bits_To_Signals(rxSignal,enc):
	dsc = {v:k for k,v in enc.items()}
	return list(map(lambda t: dsc[t],rxSignal))


def bitRepresentation(tStart,tEnd,fs,n,bits):
	time = np.arange(tStart,tEnd,1/(fs*n))
	l =[]
	for i in range(len(bits)):
		for j in range(n):
			l.append(int(bits[i][j]))
	return time,l[:len(time)]
#Question 3 

fs = 100
tStart,tEnd = 0,2*np.pi
nBits = 12  # no. of bits used for encoding
t=np.arange(tStart,tEnd,1/fs)
x=8*np.sin(t)


enc,bits,x_q=Samples_To_Bits(8,nBits,copy.deepcopy(x))

rxS=Bits_To_Signals(bits,enc)

plt.subplot(2,2,1)
plt.title("Original Signal")
plt.plot(t,x,'-c',label="8*Sin(t)")
plt.legend()
plt.grid()

plt.subplot(2,2,2)
plt.title("Quantized Signal")
plt.plot(t,x_q,'-.b',label="Tx")
plt.legend()
plt.grid()

plt.subplot(2,2,3)
plt.title("Received Quantized Signal")
plt.plot(t,rxS,'-r',label="Rx")
plt.legend()
plt.grid()

plt.subplot(2,2,4)
plt.title("Bit Representation")
bitPlot = bitRepresentation(tStart,tEnd,fs,nBits,bits)
plt.stem(bitPlot[0],bitPlot[1])


plt.show()

