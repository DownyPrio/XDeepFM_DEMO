import xDeepFM.CIN_model as CIN
import xDeepFM.DNN_model as DNN
import xDeepFM.LR_model as LR
import xDeepFM.embedding_layer as emd
from xDeepFM import *
import numpy as np

class ModelSequence():
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
        LR_res=LR.LR_model().predict(self.flattenFeature(inputSet))
        # print(LR_res.shape)
        emdFeature=emd.embedding_layer(4).embedding(inputSet)
        # print(emdFeature)
        # print(emdFeature.reshape(emdFeature.shape[0],emdFeature.shape[2]))
        CIN_res=CIN.CIN_model().predict(emdFeature.reshape(emdFeature.shape[0],emdFeature.shape[2]))
        # print(CIN_res.shape)
        DNN_res=DNN.DNN_model().predict(emdFeature.reshape(1,-1))
        # print(DNN_res.shape)
        outSet=[]
        # print(LR_res)
        # print(CIN_res)
        # print(DNN_res)
        outSet.append(LR_res)
        outSet.append(CIN_res)
        outSet.append(DNN_res)
        flatten_vector=np.array([self.flattenOut(outSet)])
        #print(flatten_vector.shape)
        W=np.zeros((flatten_vector.shape[1],1))+1
        B=1
        predictResult=Act_func.Activation("sigmoid").func(np.matmul(flatten_vector,W)+B)
        print("xDeepFM model predict results is:{}".format(str(predictResult[0][0])))
        return predictResult[0][0]

    def fit(self,feature,label):
        pass

