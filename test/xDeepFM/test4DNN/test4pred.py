import xDeepFM.DNN_model as DNN
import xDeepFM.LR_model as LR
import numpy as np
import xDeepFM.CIN_model as CIN
import xDeepFM.embedding_layer as emd

testset=np.array([[1,2]])
parameters=np.array([
    [[1,2],[3,4]],[1,2]

])

inputset=np.array([
    [1,1,1],
    [2,2,2],
    [3,3,3]
])
X=np.array([[1,2,3,4],
            [1,2,3,4],
            [1,2,3,4]])
X0=np.array([[1,1,1,1],
             [2,2,2,2],
             [3,3,3,3],
             [4,4,4,4]])
# X=np.zeros((3,4))+1
# X0=np.zeros((4,4))+1
input=np.array([[0,0,1,0],[0,1,1,0],[0,0,0,0],[1,0,0,0],[1,0,1,0,0,0,0]])

emdFeature=emd.embedding_layer(fieldD=4).embedding(input)
#print(DNN.DNN_model(1,2).predict(emdFeature))
#print(CIN.CIN_model(2,3).predict(emdFeature))
def flattenFeature(outSet):
    flatten_list=[]
    for each in outSet:
        for elements in each:
            flatten_list.append(elements)
    print(flatten_list)
    return np.array([flatten_list])
print(LR.LR_model().predict(flattenFeature(input)).shape)
# print(LR.LR_model().predict(testset))
# res_map=CIN.CIN_model(2,3).divide_col(X,X0)
# print(res_map)
# print(len(res_map))
# print(res_map[0])
# res_map=np.array(res_map)
# filterset=np.array([[[1,0,0,0],
#                     [0,1,0,0],
#                     [0,0,1,0]
#                     ],[[0,0,0,1],
#                        [0,0,1,0],
#                        [0,1,0,0]]])
# print("****************_123")
# filter_res=CIN.CIN_model(2,3).predict(inputset=inputset)
# print(filter_res)
# print(len(filter_res))
