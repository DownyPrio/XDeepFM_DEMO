import numpy as np
from xDeepFM import *
import xDeepFM.Activation as Act_func
class DNN_model(object):
    def __init__(self):
        self.layerList=[]
        self.Weights=[]
        self.biase=[]
        return
    def append(self,DNN_Layer):
        self.layerList.append(DNN_Layer)
    def build_model(self,inputData):
        if len(self.layerList)==0:
            print("no layer is detected！")
            return
        else:
            for index in range(len(self.layerList)):
                if index==0:
                    w_map=np.zeros((inputData.shape[1],self.layerList[index].nodes))+1
                    b_map=np.zeros((1,self.layerList[index].nodes))+1
                else:
                    w_map=np.zeros((self.Weights[-1].shape[1],self.layerList[index].nodes))+1
                    b_map=np.zeros((1,self.layerList[index].nodes))+1
                self.Weights.append(w_map)
                self.biase.append(b_map)
            # ow_map=np.zeros((self.layerList[-1].nodes,1))+1
            # ob_map=np.zeros((1,1))+1
            # self.Weights.append(ow_map)
            # self.biase.append(ob_map)
        self.layer_depth=len(self.layerList)
        print("-----The number of layer is:")
        print(self.layer_depth)
        print("-----DNN model is built successfully!------")
        print("Built frame weights shape is:")
        for each in self.Weights:
            print(each.shape)

    def predict(self,inputData):
        self.build_model(inputData)
        dataTensors_list=[inputData]#dataTensors_list=[inputData,*] *为每层计算结果
        dataTensors=inputData
        for index in range(self.layer_depth):
            activation_function_mode=self.layerList[index].activation
            print("-----Caculate the No.{} layer result-------".format(index+1))
            dataTensors=np.matmul(dataTensors[-1],self.Weights[index])+self.biase[index]
            dataTensors=activation_function(activation_function_mode,).activation(dataTensors)
            dataTensors_list.append(dataTensors)
            print("results:{}".format(dataTensors))
            print("-------------------------------------------")
        print("DNN model predict results is:{}".format(dataTensors[0][0]))




#DNN全连接层对象
class DNN_Layer(object):
    def __init__(self,nodes,activation):
        self.nodes=nodes
        self.activation=activation
        return

class activation_function(object):
    def __init__(self,mode):
        self.mode=mode
        return
    def activation(self,input):
        if self.mode=="relu":
            return self.relu(input)
        elif self.mode=="tanh":
            return self.tanh(input)
        elif self.mode=="sigmoid":
            return self.sigmoid(input)
        else:
            print("no such activation functions!")
    def relu(self,input):
        return (np.abs(input)+input)/2
    def tanh(self,input):
        return np.tanh(input)
    def sigmoid(self,input):
        return 1/(1+np.exp(-1*input))




