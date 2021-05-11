import matplotlib.pyplot as plt
import numpy as np
from scipy import signal 

def getKf(beta,Am,fm):
	return beta*fm/Am

def DiscreteIntegration(x,fs):
	return np.cumsum(x)/fs

def FrequencyModulateBeta(x,t,beta,fc,fs,fm):
	return np.cos(2*np.pi*fc*t+2*np.pi*getKf(beta,max(x),fm)*DiscreteIntegration(x,fs))
def FrequencyModulateKf(x,t,kf,fc,fs):
	return np.cos(2*np.pi*t*fc+2*np.pi*kf*DiscreteIntegration(x,fs)) 

def PhaseModulate(x,t,kp,fc,fs):
	return np.cos(2*np.pi*t*fc + kp*x)


plt.subplots_adjust(left=0.04,bottom=0.09,right=0.98,top=0.92,wspace=0.12,hspace=0.20)

#Q3
dt = 10**(-5)
time = np.arange(0,0.1,dt)
#Message signal
fm =50
x = 5*np.cos(2*np.pi*fm*time)

plt.subplot(241)
fc =1000
fm1 = FrequencyModulateBeta(x,time,3,fc,int(1/dt),fm)
plt.title("Q3(FM Wave) with B=3")
plt.plot(time,fm1)
plt.grid()
plt.xlabel("Time(t)->")

plt.subplot(242)
fm2 = FrequencyModulateBeta(x,time,5,fc,int(1/dt),fm)
plt.title("Q3(FM Wave) with B=5")
plt.plot(time,fm2)
plt.grid()
plt.xlabel("Time(t)->")

plt.subplot(243)
pm1 = PhaseModulate(x,time,np.pi/2,fc,int(1/dt))
plt.title("Q4(PM Wave) Kp=pi/2")
plt.plot(time,pm1)
plt.grid()
plt.xlabel("Time(t)->")

plt.subplot(244)
pm2 = PhaseModulate(x,time,np.pi/4,fc,int(1/dt))
plt.title("Q4(PM Wave) Kp=pi/4")
plt.plot(time,pm1)
plt.grid()
plt.xlabel("Time(t)->")


#Saw Tooth Generation
t = np.arange(0,0.2,dt)
sfm=int(1/10**-3)
sawSignal = signal.sawtooth(2*np.pi*sfm*t)

#Variables 
kf = 2000*np.pi
kp1 = np.pi/2
kp2= np.pi*3/2
sFc = 10**6

#Freq. Modulation of Sawtooth
plt.subplot(245)
sawFm = FrequencyModulateKf(sawSignal,t,kf,sFc,int(1/dt))
plt.title('Frequency modulated Sawtooth kf=2000*pi')
plt.plot(t,sawFm)
plt.grid()
plt.xlabel("Time(t)->")

#Phase Modulation of Sawtooth
plt.subplot(246)
sawPm1 = PhaseModulate(sawSignal,t,kp1,sFc,int(1/dt))
plt.title('Phase modulated Sawtooth kp=pi/2')
plt.plot(t,sawPm1)
plt.grid()
plt.xlabel("Time(t)->")

plt.subplot(247)
sawPm2 = PhaseModulate(sawSignal,t,kp2,sFc,int(1/dt))
plt.title('Phase modulated Sawtooth kp=3*pi/2')
plt.plot(t,sawPm2)
plt.grid()
plt.xlabel("Time(t)->")
plt.show()
