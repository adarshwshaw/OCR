import os
#import cv2
import random
import pickle
import math
#font='../char74k/EnglishFnt/English/Fnt/'
font='../caps/EnglishFnt/English/Fnt/'
i=-1
filenames=[]
for root,files,dir in os.walk(font):
    i += 1
    if i==0:
        continue;
    random.shuffle(dir)
    #print(dir)
    for d in dir:
        path = root+'/'+d
        filenames.append(path);

random.shuffle(filenames)
len=math.floor(0.8*len(filenames))
train=filenames[:len]
labels=[]
for f in filenames:
    l=int(f[-13:-10])
    labels.append(l)
traindata = open("traindata.pickle","wb")
pickle.dump(train,traindata)
traindata.close()
traindataLabels = open("traindatalabels.pickle","wb")
pickle.dump(labels[:len],traindataLabels)
traindataLabels.close()
testdata=open("testdata.pickle","wb")
pickle.dump(filenames[len:],testdata)
testdata.close()
testdatalabels=open("testdatalables.pickle","wb")
pickle.dump(labels[len:],testdatalabels)
testdatalabels.close()
print("done")
