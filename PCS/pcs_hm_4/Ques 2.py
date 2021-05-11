import matplotlib.pyplot as plt, numpy as np , math, scipy.signal as signal,copy

# Question 2 
plt.subplots_adjust(left=0.05,bottom=0.06,right=0.96,top=0.94,wspace=0.18,hspace=0.5)
def Quantize(mp,dv,x): # mp , stepsize , input array (x) , linear Uniform quantization
	l=(np.arange(-mp,mp,dv)+dv/2)
	for i in range(len(x)):
		dist = np.absolute(l-x[i])
		index = np.where(dist==min(dist))[0][0]
		x[i]= l[index]
	return x


time = np.arange(0,3,0.001)
sawtooth = 2*signal.sawtooth(2*np.pi*time,1)
triangle = 2*signal.sawtooth(2*np.pi*time,0.5)
#----Step Size: 0.2
saw_q1 = Quantize(2,0.2,copy.deepcopy(sawtooth))
tri_q1 = Quantize(2,0.2,copy.deepcopy(triangle))

#----Step Size: 2
saw_q2 = Quantize(2,2,copy.deepcopy(sawtooth))
tri_q2 = Quantize(2,2,copy.deepcopy(triangle))


plt.plot(time,sawtooth,label="Sawtooth")
plt.plot(time,saw_q1,"-.r",label="Quantized Sawtooth 1")
plt.plot(time,triangle,label="Triangle")
plt.plot(time,tri_q1,'-.c',label="Quantized Triangle 1")
plt.plot(time,saw_q2,"--g",label="Quantized Sawtooth 2")
plt.plot(time,tri_q2,"--b",label="Quantized Triangle 2")

plt.legend()
plt.show()