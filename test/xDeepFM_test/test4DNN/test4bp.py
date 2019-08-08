from xDeepFM import *
import numpy as np
import random
trainSet_x1=[]
for each in range(100):
    trainSet_x1.append(np.ones((1,2))*each)
trainSet1=trainSet_x1.copy()
w=np.array([[3],[4]])
b=np.array([[1]])
labelSet=np.matmul(trainSet1,w)+b
print(trainSet1)

trainSet=np.array([[[2]],[[4]],[[6]]])
# labelSet=np.array([[3],[5],[7]])
class seq:
    def __init__(self,y,yt):
        self.theta=yt-y
def test4bp(trainSet,epoch,rate):
    model=DNN.DNN_model()
    layer_1=DNN.DNN_Layer(2,"relu")
    model.append(layer_1)
    layer_2=DNN.DNN_Layer(2,"relu")
    layer_output=DNN.DNN_Layer(1,"relu")
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
            model.paramatersUpdate(0.01,seq_1,a)
    print(res_list)
    print(model.Weights)
    print(model.biase)

test4bp(trainSet1,1,1)
print(labelSet.reshape(1,1,-1))


