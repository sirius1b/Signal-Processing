""" Lavanya Verma (2018155) - ECE Undergrad IIITD
	PCS first programming assignment, spring 2019 
"""
import numpy as np
import matplotlib.pyplot as plt
import sys,random

plt.subplots_adjust(left=0.05,bottom=0.06,right=0.96,top=0.94,wspace=0.18,hspace=0.5)
# problem 1 
def contconv(x1,x2,t1,t2,dt):
	out=[]
	time = []
	for i in range(0,len(x1)+len(x2)):
		time.append(t1+t2+i*dt)
		var = 0
		for j in range(0,len(x1)+len(x2)):
			if (j in range(0,len(x1)) and i-j in range(0,len(x2))):
				var+=x1[j]*x2[i-j]
		out.append(var)
	out = dt*np.array(out)
	return [out,time]
 	

dt = 0.01
t1 = np.arange(-2,-1,dt)
t2 = np.arange(1,3,dt)
x1 = 3*np.ones(len(t1))
x2 = 4*np.ones(len(t2))
l =  contconv(x1,x2,int(t1[0]),int(t2[0]),dt)
plt.subplot(441)
plt.plot(l[1],l[0])
plt.title('Question 1')
plt.xlabel('Time')
plt.ylabel('3I[−2,−1]*4I[1,3]')
plt.grid()

#problem 2 
def generateUt(dt,a): # a->parameter for u(t) , u(-t)
	time = np.sort(a*np.arange(1,4,dt)) 	
	ut= 1*time
	ut = np.where(a*ut<2,2,ut)
	ut = np.where(np.logical_and(a*ut>2 ,a*ut<3),-1,ut)
	ut = np.where(np.logical_and(a*ut>3 , a*ut<4),-3,ut)
	return ut,time

def generateVt(dt,a):
	time = np.sort(a*np.arange(-1,2,dt))
	vt = 1*time
	vt = np.where(a*vt<0,1,vt)
	vt = np.where(np.logical_and(a*vt>0 , a*vt<1),3,vt)
	vt = np.where(np.logical_and(a*vt>1,a*vt<2),1,vt)
	return vt,time

def signalAddtition(x1,x2,t1,t2,dt,a,b):
	t1 = np.around(t1,decimals=3)
	t2 = np.around(t2,decimals=3)
	mint = min(min(t1),min(t2))
	maxt = max(max(t1),max(t2))
	t3 = np.arange(mint,maxt,dt)
	t3 = np.around(t3,decimals=3)
	x3=[]
	for i  in range(len(t3)):
		v = 0
		if t3[i] in t1:
			v+=a*x1[np.where(t3[i]==t1)[0][0]]
		if t3[i] in t2 :
			v+=b*x2[np.where(t3[i]==t2)[0][0]]
		x3.append(v)
	x3 = np.array(x3)
	return x3,t3

#Part1
plt.subplot(442)
ut=generateUt(dt,1)
umt =generateUt(dt,-1)
plt.title("Question 2-Part 1")
plt.plot(ut[1],ut[0],label='U(t)')
plt.plot(umt[1],umt[0],label='U(-t)')
plt.legend()
plt.grid()
plt.xlabel('Time')

#Part2 
plt.subplot(443)
m1 = contconv(ut[0],umt[0],int(round(ut[1][0])),int(round(umt[1][0])),dt)
plt.plot(m1[1],m1[0],label='')
plt.xlabel('Time')
plt.title('Convolution of U(t) and U(-t).')
plt.grid()
 
	

vt = generateVt(dt,1)
vmt = generateVt(dt,-1)
st = signalAddtition(ut[0],vt[0],ut[1],vt[1],dt,1,1j)
smt = signalAddtition(umt[0],vmt[0],umt[1],vmt[1],dt,1,-1j)

#Real components
plt.subplot(444)
plt.plot(st[1],st[0].real,label='s(t)')
plt.plot(smt[1],smt[0].real,label='s*(-t)')
plt.title("Real Components of S(t) S*(-t)")
plt.xlabel('Time(t)')
plt.ylabel('Re(s)')
plt.grid()
plt.legend()

#Imag Components
plt.subplot(445)
plt.plot(st[1],st[0].imag,label='s(t)')
plt.plot(smt[1],smt[0].imag,label='s*(-t)')
plt.title("Imaginary Components of S(t) S*(-t)")
plt.xlabel('Time(t)')
plt.ylabel('Im(s)')
plt.grid()
plt.legend()

# convolution of s(t) and s*(-t)
c =  contconv(st[0],smt[0],st[1][0],smt[1][0],dt)

plt.subplot(446)
plt.plot(c[1],c[0].real)
plt.title("Real Plot")
plt.xlabel('Time(t)')
plt.grid()

plt.subplot(447)
plt.plot(c[1],c[0].imag)
plt.title("Imaginary Plot")
plt.xlabel('Time(t)')
plt.grid()
plt.subplot(448)
plt.plot(c[1],np.absolute(c[0]))
plt.title("Magnitude Plot")
plt.xlabel('Time(t)')
plt.grid()



#Part3

def generateB(l):
	b=[]
	for i in range(int(l)): 
		b.append(1 if random.randint(0,1)==1 else -1)
	return b

def generatePt(dt):
	time = np.arange(0,1,dt)
	pt = 1*time
	pt = np.where(pt<=1,1,pt)
	return pt,time

def MultirateSys(b,p,m):
	out,time = [],[]
	dt = 1/m	
	for i in range(0,len(b)+len(p)):
		time.append(i*dt)
		var = 0
		for j in range(0,100): # 1-100 is 0-99
			if (j in range(0,len(b)) and i-j*m in range(0,len(p))):
				var += b[j]*p[i-m*j]
		out.append(var)
	out = dt*np.array(out)
	time=np.array(time)
	return [out,time]

def upCos(ut,t,f,th,a):
	out=a*ut*np.cos(f*t+th)
	return out,t
def upSin(ut,t,f,th,a):
	out=a*ut*np.sin(f*t+th)
	return out,t


pt = generatePt(dt)
plt.subplot(449)
plt.plot(pt[1],pt[0])
plt.title('PT')
plt.xlabel('Time(t)->')
plt.grid()

bc=generateB(10000)
bs=generateB(10000)
uct = MultirateSys(bc,pt[0],int(1/dt))
ust = MultirateSys(bs,pt[0],int(1/dt))
Iphase = upCos(uct[0],uct[1],4*np.pi,0,1)
Qphase = upSin(ust[0],ust[1],4*np.pi,0,1)


plt.subplot(4,4,10)
plt.plot(uct[1],uct[0]) 
plt.title('Uc(t)')
plt.xlabel('Time(t)->')
plt.grid()

plt.subplot(4,4,16)
plt.plot(ust[1],ust[0]) 
plt.title('Us(t)')
plt.xlabel('Time(t)->')
plt.grid()

########## LPF
lptime = np.arange(0,0.25,dt)
lph=1*lptime
lph = np.where(lph<1,1,lph)
##########



plt.subplot(4,4,11)
upt = signalAddtition(Iphase[0],Qphase[0],Iphase[1],Qphase[1],dt,1,-1)
plt.plot(upt[1],upt[0])
# fft =np.fft.fft(lph)
# plt.plot(np.absolute(fft))
plt.title('Up(t)')
plt.xlabel('Time(t)->')
plt.grid()








o1 = upCos(upt[0],upt[1],4*np.pi,0,1)
o2 = upSin(upt[0],upt[1],4*np.pi,0,1)
vct = contconv(o1[0],lph,o1[1][0],lptime[0],dt)
vst = contconv(o2[0],lph,o2[1][0],lptime[0],dt)

plt.subplot(4,4,12)
plt.plot(vct[1],vct[0])
plt.title("Vc(t) with theta 0")
plt.grid()
plt.xlabel('Time(t)->')

plt.subplot(4,4,13)
plt.plot(vst[1],vst[0])
plt.title('Vs(t) with theta 0')
plt.grid()
plt.xlabel('Time(t)->')

o1 = upCos(upt[0],upt[1],4*np.pi,np.pi/4,1)
o2 = upSin(upt[0],upt[1],4*np.pi,np.pi/4,1)
vct = contconv(o1[0],lph,o1[1][0],lptime[0],dt)
vst = contconv(o2[0],lph,o2[1][0],lptime[0],dt)


plt.subplot(4,4,14)
plt.plot(vct[1],-vct[0])
plt.title("Vc(t) with theta pi/4")
plt.grid()
plt.xlabel('Time(t)->')

plt.subplot(4,4,15	)
plt.plot(vst[1],vst[0])
plt.title('Vs(t) with theta pi/4')
plt.grid()

plt.xlabel('Time(t)->')

plt.show()