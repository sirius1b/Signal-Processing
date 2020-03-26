""" Lavanya Verma(2018155) Programming Assigment 4
	PCS Winter 2020 IIITD
"""

import matplotlib.pyplot as plt, numpy as np , math, scipy.signal as signal,copy

plt.subplots_adjust(left=0.05,bottom=0.06,right=0.96,top=0.94,wspace=0.10,hspace=0.24)
fs=10**6 #Sampling freq

def generateCos(A,f,ph,t1,t2,dt): # amplitude, freq , phase , starting time ,finish time , time interval
	time = np.arange(t1,t2,dt) 
	cos = A*np.cos(2*np.pi*f*time+ph)
	return cos
# Question 1 
# Part 1
tStart = 0
tEnd = 0.03
x1 = generateCos(1,60,np.pi/3,tStart,tEnd,1/fs)
x2 = generateCos(1,340,np.pi/3,tStart,tEnd,1/fs)
x3 = generateCos(1,460,np.pi/3,tStart,tEnd,1/fs)
time = np.arange(tStart,tEnd,(1/fs))

plt.subplot(2,3,1)
plt.xlabel("Time(t)-->")
plt.title("x1(t)-x2(t)")
plt.plot(time,x1-x2)

plt.subplot(2,3,2)
plt.xlabel("Time(t)-->")
plt.title("x1(t)-x3(t)")
plt.plot(time,x1-x3)

plt.subplot(2,3,3)
plt.xlabel("Time(t)-->")
plt.title("x3(t)-x2(t)")
plt.plot(time,x3-x2)

#Part 2

def sampledCos(A,a,ph,t1,t2):
	arr = np.arange(t1,t2)
	res = np.cos(a*np.pi*arr+ph)
	return res


t1 = 1
t2 = 100
time = np.arange(t1,t2)
x1_n = sampledCos(5,7.3,np.pi/4,t1,t2)
x2_n = sampledCos(5,0.7,np.pi/4,t1,t2)
x3_n = sampledCos(5,0.7,-np.pi/4,t1,t2)

plt.subplot(2,3,4)
plt.title("x1[n]-x2[n]")
plt.xlabel("(n)-->")
plt.plot(time,x1_n-x2_n)

plt.subplot(2,3,5)

plt.title("x1[n]-x3[n]")
plt.xlabel("(n)-->")
plt.plot(time,x1_n-x3_n)

plt.subplot(2,3,6)
plt.title("x3[n]-x2[n]")
plt.xlabel("(n)-->")
plt.plot(time,x3_n-x2_n)





plt.show()