from xDeepFM import *
import numpy as np
import random
trainSet_x1=[]
list_p=[1]*500
print(list_p)
list_1=list(map(lambda x:x*random.randrange(1,100),list_p))
list_2=list(map(lambda x:x*random.randrange(1,100),list_p))
trainSet=np.array([list_1,list_2]).T
trainFinalSet=[]
for index in range(len(trainSet)):
    trainFinalSet.append(np.array([trainSet[index]]))


print(trainFinalSet)
w=np.array([[3],[4]])
b=np.array([[1]])
labelSet=np.matmul(trainFinalSet,w)+b
print(trainFinalSet)

class seq:
    def __init__(self,y,yt):
        self.theta=yt-y
def test4bp(trainSet,epoch,rate):
    model=DNN.DNN_model()
    layer_1=DNN.DNN_Layer(2,"none")
    model.append(layer_1)
    layer_2=DNN.DNN_Layer(2,"none")
    layer_output=DNN.DNN_Layer(1,"none")
    model.append(layer_2)
    model.append(layer_output)

    for i in range(10000):
        res_list=[]
        print("epochs:{}/10000".format(i))
        for index in range(len(trainSet)):
            result,a=model.predict(trainSet[index])
            seq_1=seq(labelSet[index][0][0],result[0][0])
            #print(result)
            res_list.append(result)
            model.paramatersUpdate(0.0000000001,seq_1,a)
    t_list=[]
    for each in trainSet:
        print(each)
        result,a=model.predict(each)
        t_list.append(result)

    print(t_list)
    print(model.Weights)
    print(model.biase)
    return t_list

t=test4bp(trainFinalSet,1,1)
print(t)
print(labelSet.reshape(1,1,-1))


