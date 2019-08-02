import xDeepFM.DNN_model as DNN
import xDeepFM.LR_model as LR
import numpy as np
import xDeepFM.CIN_model as CIN
import xDeepFM.embedding_layer as emd
from xDeepFM import *

testset=np.array([[1,2]])
parameters=np.array([
    [[1,2],[3,4]],[1,2]

])

inputset=np.array([
    [1,1,1],
    [2,2,2],
    [3,3,3]
])

input=np.array([[0,0,1,0],[0,1,1,0],[0,0,0,0],[1,0,0,0],[1,0,1,0,0,0,0]])

# emdFeature=emd.embedding_layer(fieldD=4).embedding(input)
# print(emdFeature.shape)
def flattenFeature(outSet):
    flatten_list=[]
    for each in outSet:
        for elements in each:
            flatten_list.append(elements)
    print(flatten_list)
    return np.array([flatten_list])

# emdFeature4DNN=emdFeature.reshape((1,-1))
#print(emdFeature4DNN)
# print(DNN.DNN_model(3,3).predict(emdFeature4DNN).shape)
# emdFeature4CIN=emdFeature.reshape((input.shape[0],-1))
# print(CIN.CIN_model(3,3).predict(emdFeature4CIN).shape)
res=Sequence.ModelSequence().predict(input)
#print(res)