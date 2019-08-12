from xDeepFM import *
import xDeepFM.Activation as Act_func
import numpy as np

class ModelSequence():
    def __init__(self,LR,CIN,DNN,EMD):
        self.LR_model=LR
        self.CIN_model=CIN
        self.DNN_model=DNN
        self.EMD_model=EMD
    def flattenFeature(self,outSet):
        flatten_list=[]
        for each in outSet:
            for elements in each:
                flatten_list.append(elements)
        return np.array(flatten_list)
    def flattenOut(self,outSet):
        flatten_list=[]
        for each in outSet:
            for elements in each[0]:
                flatten_list.append(elements)
        return np.array(flatten_list)
    def predict(self,inputSet):
        #输出格式均为np.array((1,n)),即[[1,2,3,4,5]]
        # print(inputSet)
        LR_res=self.LR_model.predict(CIN.CIN_model().flatten(inputSet))
        self.B=1
        self.LR_W=np.zeros((1,len(LR_res)))

        # print(LR_res.shape)
        emdFeature=self.EMD_model.embedding(inputSet)
        # print("emdFeature")
        # print(emdFeature.shape)
        # print(emdFeature.reshape(emdFeature.shape[0],emdFeature.shape[2]))
        CIN_res=self.CIN_model.predict(emdFeature)
        print("")
        print(CIN_res)
        self.CIN_W=np.zeros((CIN_res.shape[1],1))+1

        # print(CIN_res.shape)
        DNN_res=self.DNN_model.predict(emdFeature.reshape(1,-1))
        self.DNN_W=np.zeros((DNN_res.shape[1],1))+1

        # print(DNN_res.shape)
        outSet=[]
        # print(LR_res)
        # print(CIN_res)
        # print(DNN_res)
        outSet.append(LR_res)
        outSet.append(CIN_res)
        outSet.append(DNN_res)
        # print("out")
        # print(outSet)
        LR_Part=(np.matmul(LR_res,self.LR_W))[0][0]
        print("___www")
        print(CIN_res.shape)
        print(self.CIN_W.shape)
        CIN_Part=(np.matmul(CIN_res,self.CIN_W))[0][0]
        DNN_Part=(np.matmul(DNN_res,self.DNN_W))[0][0]
        predictResult=self.sigmoid(LR_Part+CIN_Part+DNN_Part+self.B)
        print("xDeepFM model predict results is:{}".format(predictResult))
        return predictResult
    def sigmoid(self,input):
        return 1/(1+np.exp(-input))
    def paramatersUpdate(self,label,result,tmp_result,N,rate):
        LR_res=tmp_result[0]
        CIN_res=tmp_result[1]
        DNN_res=tmp_result[2]
        if label==1:
            theta=(1-result)/N
        else:
            theta=(-result)/N
        self.theta=theta
        self.LR_W_delta=-rate*theta*LR_res
        self.CIN_W_delta=-rate*theta*CIN_res
        self.DNN_W_delta=-rate*theta*DNN_res
        self.B_delta=-rate*theta
    def fit(self,feature,label,epochs,learning_rate_list,optimizer_list):
        self.LR_model.fit(feature,label,epochs,learning_rate_list[0],self,optimizer_list[0])
        self.DNN_model.fit(feature,label,epochs,learning_rate_list[1],self,optimizer_list[1])
        self.CIN_model.fit(feature,label,epochs,learning_rate_list[2],self,optimizer_list[2])


