import numpy as np
import random

a=[x for x in range(50)]
b=[y for y in range(20,70)]
a=a*20
b=b*20
random.shuffle(b)
print(a)
print(b)
fSet=list(zip(a,b))
f=[]
for each in fSet:
    f.append(np.array(each))
f=np.array(f)

print(fSet)
label=[]
for index in range(1000):
    label.append(3*a[index]+4*b[index]+3)
print(label)

weights=np.ones((2,1))
biase=3
rate=0.0005
print(np.sqrt(np.abs(5**2)))
def mse(a,b):
    print(a)
    print(b)
    return np.sqrt(np.abs(np.sum(np.array(a-b))**2)/len(a))
def sgd_optimizer(f):
    global weights
    global biase
    global label
    for epoch in range(100):
        print("epoch : {}/1000".format(epoch))
        for index in range(len(label)):
            #print("...........")
            label=np.array(label).reshape((-1,1))
            ytest=np.matmul(f,weights)+biase
            # print(ytest.shape)
            # print(np.array(label).shape)
            #print(ytest.T[0][index]-label.T[0][index])
            weights-=rate*f[index].reshape((-1,1))*(ytest.T[0][index]-label.T[0][index])
            biase-=rate*(ytest[index]-label[index])
        # print("**************")
        # print("loss:")
        # #print(mse(np.matmul(f,weights)+biase,label))
        # print("**************")
    print(weights)
    print(biase)
sgd_optimizer(f)