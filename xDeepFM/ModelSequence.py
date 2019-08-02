import xDeepFM.CIN_model as CIN
import xDeepFM.DNN_model as DNN
import xDeepFM.LR_model as LR
import xDeepFM.embedding_layer as emd
import numpy as np

class ModelSequence():
    def flattenFeature(self,outSet):
        flatten_list=[]
        for each in outSet:
            for elements in each:
                flatten_list.append(elements)
        return np.array(flatten_list)

    def predict(self,inputSet):
        LR_res=LR.LR_model().predict(inputSet)
        emdFeature=emd.embedding_layer().embedding(inputSet)
        CIN_res=CIN.CIN_model().predict(emdFeature)
        DNN_res=DNN.DNN_model().predict(emdFeature)
        outSet=[]
        outSet.append(LR_res)
        outSet.append(CIN_res)
        outSet.append(DNN_res)
        flatten_vector=self.flattenFeature(outSet)
        W=np.zeros((flatten_vector.shape[1],1))+1
        B=1
        predictResult=np.matmul(flatten_vector,W)+B
        return predictResult
