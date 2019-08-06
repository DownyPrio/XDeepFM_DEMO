from xDeepFM import *
import pandas as pd
filename="data"
data=pd.read(filename)
trainSet=create(data)#训练集：（特征，标签）
#LR_part
LR_Result=LR_model.predict(trainSet)
embeddingFeature=embeddingLayer.predict(trainSet)

#CIN_part
CIN_model=CIN_model()
CIN_model.append(CIN_Layer(H))
CIN_model.append(CIN_Layer(H))
......
CIN_Result=CIN_model.predict(embeddingFeature)#包含Sum池化过程

#DNN_part
DNN_model=DNN_model()
DNN_model.append(DNN_Layer(N))
DNN_model.append(DNN_Layer(N))
.......
DNN_Result=DNN_model.predict(embeddingFeature)#Relu或tanh

#assemble
featureFlattened=[LR_Result,CIN_Result,DNN_Result]
result=SIGMOID(W*featureFlattened+B)

