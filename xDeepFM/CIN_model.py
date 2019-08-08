import numpy as np
import xDeepFM.LR_model as LR
from xDeepFM import *

class CIN_model(object):
    def __init__(self):
        self.layerList=[]
        self.Weights=[]
        self.biase=[]
        return
    def flatten(self,X):
        # print(X)
        flatten_list=[]
        for raw in X:
            for col in raw:
                flatten_list.append(col)
        return np.array([flatten_list])
    def featureInteraction(self,Xh,X0):
        XhT=Xh.T
        X0T=X0.T

        for index in range(len(X0)):
            print(XhT.shape)
            print(X0T.shape)
            print(XhT[index].reshape(-1,1))
            print(X0T[index])
            tmp_vector=np.matmul(XhT[index].reshape(-1,1),X0T[index].reshape(1,-1))
            tmp_vector=self.flatten(tmp_vector)

            if index==0:
                tmp_vector_list=tmp_vector
            else:
            #print(tmp_vector)
                tmp_vector_list=np.insert(tmp_vector_list,len(tmp_vector_list),tmp_vector,axis=0)
        return tmp_vector_list
    def sumPool(self,input):
        print(np.sum(input,axis=0))
        return np.sum(input,axis=0)
    def append(self,CIN_Layer):
        self.layerList.append(CIN_Layer)
    def build_model(self,inputData):
        if len(self.layerList)==0:
            print("no layer is detected！")
            return
        else:
            for index in range(len(self.layerList)):
                if index==0:
                    w_map=np.zeros((inputData.shape[0]*inputData.shape[0],self.layerList[0].H))+1
                else:
                    w_map=np.zeros((self.layerList[index-1].H*inputData.shape[0],self.layerList[index].H))+1
                self.Weights.append(w_map)
        self.layer_depth=len(self.layerList)
        print("-----The number of layer is:")
        print(self.layer_depth)
        print("-----CIN model is built successfully!------")
        print("Built frame weights shape is:")
        for each in self.Weights:
            print(each.shape)
    def predict(self,inputData):
        self.build_model(inputData)
        dataTensors_list=[inputData]#dataTensors_list=[inputData,*] *为每层计算结果
        dataTensors=inputData
        sumPoolList=[]
        for index in range(self.layer_depth):
            print("-----Caculate the No.{} layer result-------".format(index+1))
            tmp_vector=self.featureInteraction(dataTensors_list[index],dataTensors_list[0])
            print(tmp_vector.shape)
            print(self.Weights[index])
            dataTensors=np.matmul(tmp_vector,self.Weights[index])
            dataTensors_list.append(dataTensors.T)
            sumFeature=self.sumPool(dataTensors)
            sumPoolList.append(sumFeature)
            print("results:{}".format(dataTensors))
            print("-------------------------------------------")

        resultVector=self.flatten(sumPoolList)
        print("CIN model predict results is:{}".format(resultVector))
        return resultVector

    def paramatersUpdate(self,rate,seq,inputFeature):
        pass



class CIN_Layer(object):
    def __init__(self,H):
        self.H=H
        return


