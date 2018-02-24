import cv2
import pickle as pkl
import numpy as np

trn=open('./traindata.pickle','rb')
traindata= pkl.load(trn)
trn.close();
trn=open('./traindatalabels.pickle','rb')
traindatalabels = pkl.load(trn)
trn.close()
tst=open('./testdata.pickle','rb')
testdata=pkl.load(tst)
tst.close()
tst=open('./testdatalables.pickle','rb')
testdatalabels=pkl.load(tst)
tst.close()

def one_hot_labels(index):
    label= np.zeros(36)
    #print(i)
    label[index-1]=1
    return label

def load_batch(start=0,end=0,testing=0):
    x=[]
    y=[]
    global count
    if testing == 1:
        data=testdata;
        labels=testdatalabels
        batch_size=len(data)
        end=start+batch_size
    else:
        data=traindata
        labels=traindatalabels
    i=start
    while i < end:
        try:
            g=data[i]
        except IndexError:
            break
        img = cv2.imread(data[i])
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (20, 20))
        img = np.array(img)
        img = img.flatten()
        x.append(img)
        y.append(one_hot_labels(labels[i]))
        i += 1
    x=np.array(x)
    y=np.array(y)
    return x,y

'''x,y=load_batch(0,50)
print(x)
cv2.imshow('sdf',x[0].reshape(20,20))
cv2.waitKey(0)
cv2.destroyAllWindows()'''