"""
Author: Lavanya Verma(2018155)
email : lavanya18155@iiitd.ac.in
Solution of Q3, CV Assignment 2. 
CV @ IIITD Winter 2021
"""

from Q2 import *
from scipy import integrate

def sol(nu_f, nu_b , std_f, std_b):
    a = (nu_b*std_f**2 - nu_f*std_b**2)
    b = std_f**2 - std_b**2
    c = std_b*std_f*np.sqrt((nu_f - nu_b)**2 - 2*b*(np.log(std_b) - np.log(std_f))) 
    z1 , z2 = (a + c)/b, (a - c)/b
    z = z1 if nu_b < z1 and z1 < nu_f else z2 
    # print(z1,z2)
    return z

def seperation_measure(sal):
    sal = (sal - sal.min())/(sal.max() - sal.min())
    ret, th = cv2.threshold((sal*(2**16)).astype(np.uint16), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU); th = th == 1
    back = np.ma.array(sal,mask = th)
    forg = np.ma.array(sal,mask = ~th)
    nu_f = forg.mean(); std_f = forg.std();  nu_b = back.mean(); std_b = back.std()
    df = lambda x: np.exp( - ((x - nu_f)/ std_f)**2)/(std_f *np.sqrt(2*np.pi))
    db = lambda x: np.exp( - ((x - nu_b)/ std_b)**2)/(std_b *np.sqrt(2*np.pi))
    z_star = sol(nu_f,nu_b,std_f,std_b) 
    L = integrate.quad(df , 0 , z_star)[0] + integrate.quad(db , z_star , 1)[1]
    gamma = 16
    phi = 1 / (1 + np.log10(1 + gamma * L))
    x = np.arange(0, 1 , 0.005)
    return phi

def concentration_measure(sal):
    sal = (sal - sal.min())/(sal.max() - sal.min())
    ret, th = cv2.threshold((sal*(2**16)).astype(np.uint16), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU); 
    forg = th
    num_labels, labels = cv2.connectedComponents((forg.astype(np.uint8)))
    Cus = []
    for i in range(num_labels):
        Cus.append((labels == i).sum())

    # plt.imshow(forg)
    # plt.show()

    Cus = np.array(Cus)
    Cus = Cus/Cus.sum()
    psi = max(Cus) + (1  - max(Cus))/(num_labels)
    return psi

if __name__ == '__main__':
    image1 = np.asarray(Image.open('1558014721_E7jyWs_iiit_d.jpg'))
    n_segments = 100
    K = 8
    sal1,mask,K = contrastcue(image1,K,n_segments)
    sal1 = (sal1 - sal1.min())/(sal1.max() - sal1.min())
    phi = seperation_measure(sal1) 
    psi = concentration_measure(sal1)
    w1 = phi*psi
    print('Contrast Cue Score: %f phi: %f psi: %f'%(w1,phi,psi))
    sal2 = spatialcue(image1,K,mask)
    sal2 = (sal2 - sal2.min())/(sal2.max() - sal2.min())
    phi = seperation_measure(sal2) 
    psi = concentration_measure(sal2)
    w2 = phi*psi
    print('Spatial Cue Score: %f phi: %f psi: %f'%(w2,phi,psi))
    plt.imshow(w1*sal1 + w2*sal2, cmap = cm.gray)
    plt.title("Weighted Sum(Saliency Map)")
    plt.axis('off')
    plt.show()
