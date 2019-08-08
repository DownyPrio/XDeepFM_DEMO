import numpy as np
from xDeepFM import *
import xDeepFM.Activation as Act_func
class DNN_model(object):
    def __init__(self):
        self.layerList=[]
        self.Weights=[]
        self.biase=[]
        self.built=False
        return

    def append(self,DNN_Layer):
        self.layerList.append(DNN_Layer)

    def build_model(self,inputData):
        # print("inputData4dnn")
        # print(inputData)
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
        # print("-----The number of layer is:")
        # print(self.layer_depth)
        # print("-----DNN model is built successfully!------")
        # print("Built frame weights shape is:")
        # for each in self.Weights:
        #     print(each.shape)
        self.built=True
    def predict(self,inputData):
        if not self.built:
            self.build_model(inputData)
        dataTensors_list=[inputData]#dataTensors_list=[inputData,*] *为每层计算结果
        dataTensors=inputData
        for index in range(self.layer_depth):
            activation_function_mode=self.layerList[index].activation
            # print("-----Caculate the No.{} layer result-------".format(index+1))
            dataTensors=np.matmul(dataTensors[-1],self.Weights[index])+self.biase[index]
            dataTensors=activation_function(activation_function_mode,).activation(dataTensors)
            dataTensors_list.append(dataTensors)
        #     print("results:{}".format(dataTensors))
        #     print("-------------------------------------------")
        # print("DNN model predict results is:{}".format(dataTensors))
        return dataTensors,dataTensors_list
    def delta_process(self,list1,list2):
        result_list=[]
        for index in range(len(list1)):
            result_list.append(list1[index]-list2[index])
        return result_list
    def mul_process(self,rate,list1):
        result_list=[]
        for index in range(len(list1)):
            result_list.append(rate*list1[index])
        return result_list
    def trans(self,list1):
        for index in range(len(list1)):
            list1[index]=list1[index].T
    #输入：学习率，带有theta项的sequence对象，暂存数据
    def paramatersUpdate(self,rate,seq,dataTensors_list):
        W_delta_list=[]
        B_delta_list=[]
        theta_list=[np.array(seq.theta).reshape((1,1))]
        for index in range(len(self.layerList)-1,-1,-1):
            # print(self.Weights[index].T.shape)
            # print(theta_list[-1].shape)
            # print(len(dataTensors_list))
            product_mul=np.matmul(self.Weights[index],theta_list[-1])
            #product_star=activation_function("relu").relu_derivative(dataTensors_list[index]).reshape((-1,1))#*self.Weights[index]
            product_star=dataTensors_list[index].reshape((-1,1))
            # print(product_mul)
            # print(product_star)
            theta=product_star*product_mul
            #theta=np.dot(product_mul,product_star)
            theta_last=theta_list[-1]
            # print("....................../////////")
            # print(product_mul)
            # print(product_star)
            # print(theta_last)
            # print(dataTensors_list[index])
            # print("......................////////////")
            W_delta_list.append(np.matmul(theta_last,dataTensors_list[index]))
            self.trans(W_delta_list)
            B_delta_list.append(theta_last.reshape((1,-1)))
            theta_list.append(theta)
        W_delta_list.reverse()
        B_delta_list.reverse()
        # print(self.biase)
        # print(B_delta_list)
        self.Weights=self.delta_process(self.Weights,self.mul_process(rate,W_delta_list))
        self.biase=self.delta_process(self.biase,self.mul_process(rate,B_delta_list))

    def fit(self,trainSet,labelSet,epochs,learning_rate,seq,optimizer="sgd"):
        if optimizer=="sgd":
            for i in range(epochs):
                print("epochs:{}/{}".format(i,epochs))
                for index in range(len(trainSet)):
                    result,dataTensors_list=self.predict(trainSet[index])
                    seq_1=seq(labelSet[index][0][0],result[0][0])
                    self.paramatersUpdate(learning_rate,seq_1,dataTensors_list)
            return
        elif optimizer=="batch_sgd":
            return





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
        elif self.mode=="none":
            return input
        else:
            print("no such activation functions!")
    def relu(self,input):
        return (np.abs(input)+input)/2
    def tanh(self,input):
        return np.tanh(input)
    def sigmoid(self,input):
        return 1/(1+np.exp(-1*input))
    def relu_derivative(self,input):
        def relu(tmp):
            if tmp==0:
                return 0
            else:
                return 1
        return np.array(list(map(relu,input[0]))).reshape((1,-1))





