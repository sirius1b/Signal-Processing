from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import Dataset, DataLoader
from mlxtend.data import loadlocal_mnist
from torch.autograd import Variable
import torch
import cv2

def TSS_Segment(data):
    hist = np.zeros((256))
    for i in range(256):
        hist[i] += (data == i).sum()
    hist = hist/hist.sum()
    thre = -1 ; val = np.inf
    for i  in range(1,256):
        w1 = hist[:i].sum()
        w2 = hist[i:].sum()
        s1_mean = np.array([ii*hist[ii]/w1 for ii in range(i)]).sum()
        s2_mean = np.array([ii*hist[ii]/w2 for ii in range(i,256)]).sum()
        s1 = np.square([ii - s1_mean for ii in range(i)]).sum()
        s2 = np.square([ii - s2_mean for ii in range(i,256)]).sum()
        if (s1 + s2 < val):
            val = s1+s2
            thre = i
    mask = data >= thre
    # print(thre)
    return mask


def minCIRCLE(img):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[-2]
    (x,y), r = cv2.minEnclosingCircle(cnts[0])
    return x,y,r


class DataAugment(Dataset):
    def __init__(self,root,type, train = True):
        if train:       
            self.x, self.y = loadlocal_mnist(images_path = root + 'train-images-idx3-ubyte', labels_path = root +  'train-labels-idx1-ubyte')
        else:
            self.x, self.y = loadlocal_mnist(images_path = root + 't10k-images-idx3-ubyte', labels_path = root + 't10k-labels-idx1-ubyte')   
        self.type = type
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self,index):
        img = self.x[index].reshape(28,28)
        label = self.y[index]
        mask = TSS_Segment(img)
        # print(mask.dtype,mask.max(), mask.min())
        if self.type == 1:
            img = img.reshape(1,28,28)
            mask = mask.reshape(1,28,28)
            data = {'x':torch.tensor((img/255).astype(np.float32)),
                    'y_label': torch.tensor(label),
                 'y_mask': torch.tensor(mask.astype(np.float32))}
        elif self.type == 2:
            cx, cy, r = minCIRCLE((mask).astype(np.uint8))
            img = img.reshape(1,28,28)
            mask = mask.reshape(1,28,28)
            data = {'x':torch.tensor((img/255).astype(np.float32)),
                 'y_label': torch.tensor(label,dtype=torch.long),
                 'y_mask': torch.tensor(mask.astype(np.float32)),
                 'y_cx': torch.tensor(cx/28,dtype= torch.float32),
                 'y_cy':torch.tensor(cy/28,dtype= torch.float32),
                 'y_r': torch.tensor(r/28,dtype= torch.float32)}
        return data



class DataAugment3(Dataset):
    def __init__(self,root, train = True):
        if train:       
            self.x, self.y = loadlocal_mnist(images_path = root + 'train-images-idx3-ubyte', labels_path = root +  'train-labels-idx1-ubyte')
        else:
            self.x, self.y = loadlocal_mnist(images_path = root + 't10k-images-idx3-ubyte', labels_path = root + 't10k-labels-idx1-ubyte')   
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self,idx0):
        idx1 = int(len(self)*np.random.random())
        idx2 = int(len(self)*np.random.random())
        idx3 = int(len(self)*np.random.random())
        idxs = np.random.permutation([idx0, idx1, idx2, idx3])

        bIMG = np.zeros((56,56))
        bSEG_MASK = np.zeros((56,56))

        img = self.x[idxs[0]].reshape(28,28)
        mask = TSS_Segment(img)
        s_mask = np.ones((28,28))*10
        s_mask[mask] = self.y[idxs[0]] 
        bIMG[0:28,0:28] = img       
        bSEG_MASK[0:28,0:28] = s_mask

        img = self.x[idxs[1]].reshape(28,28)
        mask = TSS_Segment(img)
        s_mask = np.ones((28,28))*10
        s_mask[mask] = self.y[idxs[1]] 
        bIMG[0:28,28:56] = img       
        bSEG_MASK[0:28,28:56] = s_mask
        
        img = self.x[idxs[2]].reshape(28,28)
        mask = TSS_Segment(img)
        s_mask = np.ones((28,28))*10
        s_mask[mask] = self.y[idxs[2]] 
        bIMG[28:56,0:28] = img       
        bSEG_MASK[28:56,0:28] = s_mask

        img = self.x[idxs[3]].reshape(28,28)
        mask = TSS_Segment(img)
        s_mask = np.ones((28,28))*10
        s_mask[mask] = self.y[idxs[3]] 
        bIMG[28:56,28:56] = img       
        bSEG_MASK[28:56,28:56] = s_mask

        data = {
        	'x': torch.tensor(bIMG.reshape(1,56,56)/255,dtype = torch.float),
        	'seg_mask': torch.tensor(bSEG_MASK.reshape(56,56),dtype = torch.long)
        }
        return data

def JSim1(model,test_loader,device):
    average_similarity = 0
    model.eval()
    with torch.no_grad():
        for idx,data in enumerate(test_loader):
            inpt = data['x'].to(device)
            mask = data['y_mask'].to(device)
            output = model(inpt).cpu()
            for i in range(len(data['x'])):
                a = data['y_mask'][i].numpy()
                b = (output[i].detach() >0.5).numpy()
                sim = np.logical_and(a == 1, b ==1 ).sum()/np.logical_or(a == 1, b ==1 ).sum()
                average_similarity += sim
        average_similarity /= len(test_loader.dataset)
    return average_similarity
            
def getMask(cx,cy,r):
    con = np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            if (np.linalg.norm([cx - j,cy - i])) < r:
                con[i,j] = True
    return con

def JSim2(model, test_loader,device):
    average_similarity = 0
    model.eval()
    with torch.no_grad():
        for idx,data in enumerate(test_loader):
            inpt = data['x'].to(device)
            r = data['y_r'].to(device); cx = data['y_cx'].to(device); cy = data['y_cy'].to(device)
            label = data['y_label'].to(device)
            o1, o2 = model(inpt)
            res = o1.argmax(dim = 1) == label
            for i in range(len(data['x'])):
                a = getMask(cx[i]*28,cy[i]*28,r[i]*28)
                b = getMask(o2[i,1]*28,o2[i,2]*28,o2[i,0]*28)
                sim = res[i]*np.logical_and(a , b).sum()/np.logical_or(a , b).sum()
                average_similarity += sim
#                 return sim
        average_similarity /= len(test_loader.dataset)
    return average_similarity

def JSim3(model,test_loader,device):
    average_similarity = 0
    model.eval()
    with torch.no_grad():
        for idx,data in enumerate(test_loader):
            print("{}/{}".format(idx,len(test_loader)))
            inpt = data['x'].to(device)
            seg_mask = data['seg_mask'].to(device)
            output = torch.exp(model(inpt).cpu()).argmax(dim = 1)
            seg_mask = seg_mask.cpu()
            for i in range(len(data['x'])):
                class_score = [0]*10
                m1 = output[i,:,:]
                m2 = seg_mask[i,:,:]
                for j in range(10):
                    a = float(np.logical_and(m1==j , m2==j).sum())
                    b = float(np.logical_or(m1 ==j , m2==j).sum())
                    if (b != 0):
                        class_score[j] = a/b
                class_score = np.array(class_score)
                average_similarity += np.ma.masked_equal(class_score,0).mean()
        average_similarity /= len(test_loader.dataset)
    return average_similarity