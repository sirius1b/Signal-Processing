""" Lavanya Verma(2018155) Programming Question
	Winter 2020 PCS, Assignment 2
	IIITD 
"""
import matplotlib.pyplot as plt, numpy as np,math



def signalAddtition(x1,x2,t1,t2,dt,a,b):
        ds = int(np.absolute(math.log10(dt)))
        t1 = np.around(t1,decimals=ds)
        t2 = np.around(t2,decimals=ds)
        mint = min(min(t1),min(t2))
        maxt = max(max(t1),max(t2))
        t3 = np.arange(mint,maxt,dt)
        t3 = np.around(t3,decimals=ds)
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

dt = 0.0000001
#Question 5
plt.subplots_adjust(left=0.04,bottom=0.06,right=0.98,top=0.94,wspace=0.24,hspace=0.35)

def generateCos(A,f,ph,t1,t2,dt): # amplitude, freq , phase , starting time ,finish time , time interval
	time = np.arange(t1,t2,dt) 
	cos = A*np.cos(2*np.pi*f*time+ph)
	return cos,time

def generateSin(A,f,ph,t1,t2,dt): # amplitude, freq , phase , starting time ,finish time , time interval
	time = np.arange(t1,t2,dt) 
	sin = A*np.sin(2*np.pi*f*time+ph)
	return sin,time
timeS=0
timeE=0.0002
#Part 1 DSB-SC signal
plt.subplot(251)
mt = generateCos(1,5000,0,timeS,timeE,dt)
carrier = generateCos(1,500000,0,timeS,timeE,dt)
Udsb = mt[0]*carrier[0]
plt.ylabel('DSB-SC')
plt.title('Question 5 Part 1')
plt.xlabel('Time(t)-->')
plt.grid()
plt.plot(mt[1],Udsb)

plt.subplot(252)
plt.ylabel('m(t)-->')
plt.title('Message Signal ')
plt.xlabel('Time(t)-->')
plt.grid()
plt.plot(mt[1],mt[0])



#Part 2 Conventional AM Scenario
plt.subplot(253)
# ac = int(input("Conventional Am AC Value: ")) 
Uam1 = 1*carrier[0]+Udsb
plt.ylabel('DSB-AM Modulated Wave')
plt.title('Question 5 Part 2 with A=1')
plt.xlabel('Time(t)-->')
plt.grid()
plt.plot(mt[1],Uam1)


plt.subplot(254)
# ac = int(input("Conventional Am AC Value: ")) 
Uam2 = 2*carrier[0]+Udsb
plt.ylabel('DSB-AM Modulated Wave')
plt.title('Question 5 Part 2 with A=2')
plt.xlabel('Time(t)-->')
plt.grid()
plt.plot(mt[1],Uam2)

plt.subplot(255)
Uam3 = 0.5*carrier[0]+Udsb
plt.ylabel('DSB-AM Modulated Wave')
plt.title('Question 5 Part 2 with A=0.5')
plt.xlabel('Time(t)-->')
plt.grid()
plt.plot(mt[1],Uam3)





#Question 6
#Part 1
plt.subplot(256)
mSin = generateSin(1,5000,0,timeS,timeE,dt)
mCos = generateCos(1,5000,0,timeS,timeE,dt)
carrierCos = generateCos(1,500000,0,timeS,timeE,dt)
carrierSin = generateSin(1,500000,0,timeS,timeE,dt)
I=mCos[0]*carrierCos[0]
Q=mSin[0]*carrierSin[0]
Uusb=signalAddtition(I,Q,mSin[1],mCos[1],dt,1,-1)

plt.ylabel('SSB-SC')
plt.title('Question 6 Part 1')
plt.xlabel('Time(t)-->')
plt.grid()
plt.plot(Uusb[1],Uusb[0])


plt.subplot(257)
Uusbam1 = Uusb[0]+carrier[0]*2

plt.ylabel('SSB-AM Modulated Wave')
plt.title('Question 5 Part 2 with A=2')
plt.xlabel('Time(t)-->')
plt.grid()
plt.plot(Uusb[1],Uusbam1)

plt.subplot(258)
Uusbam2 = Uusb[0]+carrier[0]*1

plt.ylabel('SSB-AM Modulated Wave')
plt.title('Question 5 Part 2 with A=1')
plt.xlabel('Time(t)-->')
plt.grid()
plt.plot(Uusb[1],Uusbam2)

plt.subplot(259)
Uusbam3 = Uusb[0]+carrier[0]*0.5

plt.ylabel('SSB-AM Modulated Wave')
plt.title('Question 5 Part 2 with A=0.5')
plt.xlabel('Time(t)-->')
plt.grid()
plt.plot(Uusb[1],Uusbam3)


plt.show()

