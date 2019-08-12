import numpy as np
import xDeepFM.LR_model as LR
from xDeepFM import *

class CIN_model(object):
    def __init__(self):
        self.layerList=[]
        self.Weights=[]
        self.biase=[]
        self.built=False
        self.theta_list=[]
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
        self.theta_list.append(CIN_Layer.CIN_theta)
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
        self.built=True
    def predict(self,inputData):
        if not self.built:
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
        return resultVector,dataTensors_list
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
    #参数更新部分

    def paramatersUpdate(self,rate,seq,inputFeature):
        real_theta=[self.theta_list[-1]]
        W_delta_list=[seq.CIN_W[-1]*real_theta[-1]]
        for index in range(self.layer_depth-1,-1,-1):
            real_theta.append(self.theta_list[index]+np.matmul(real_theta[-1],self.Weights[index+1]))
            W_delta_list.append(seq.CIN_W[index]*real_theta[-1])
        W_delta_list.reverse()
        self.Weights=self.delta_process(self.Weights,self.mul_process(rate,W_delta_list))


    def fit(self,trainSet,labelSet,epochs,learning_rate,seq,optimizer="sgd"):
        if optimizer=="sgd":
            for i in range(epochs):
                print("epochs:{}/{}".format(i,epochs))
                for index in range(len(trainSet)):
                    result,dataTensors_list=self.predict(trainSet[index])
                    seq_l=seq(labelSet[index][0][0],result[0][0])
                    self.paramatersUpdate(learning_rate,seq_l,dataTensors_list)
            return
        elif optimizer=="batch_sgd":
            return

class CIN_Layer(object):
    def __init__(self,H):
        self.H=H
        return
    def cal_theta(self,rate,seq,index,X_product,W_this):
        #X_product is (D,H0*Hn-1)
        #W_this is (H0*Hn-1,Hn)
        #sum_res=sum_pool result
        theta=seq.theta
        self.CIN_theta=np.sum(X_product.T,axis=1)






