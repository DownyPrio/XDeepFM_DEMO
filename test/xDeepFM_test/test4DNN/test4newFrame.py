from xDeepFM import *
#import numpy as np

#LR_model test
input=np.array([[1,2,3,4,5]])
def test4LR(input):
    LR_result=LR.LR_model().predict(input)
    print(LR_result)
    return LR_result
#test4LR(input)

#DNN_model test
def test4DNNmodel():
    model=DNN.DNN_model()
    single_layer=DNN.DNN_Layer(3,"relu")
    model.append(single_layer)
    print(model.layerList)
"""
DNN_LAYER test
"""
def test4DNNLayer(input):
    model_layer=DNN.DNN_model()
    model_layer.append(DNN.DNN_Layer(3,"relu"))
    model_layer.append(DNN.DNN_Layer(4,"relu"))
    model_layer.append(DNN.DNN_Layer(5,"relu"))
    model_layer.append(DNN.DNN_Layer(1,"sigmoid"))
    #model_layer.build_model(input)
    model_layer.predict(input)
    print(DNN.activation_function("sigmoid").activation(30))
#test4DNNLayer(input)

#embedding test
inputSparse=np.array([[1,2,3],[1,2],[1]])
def test4EMD(inputSparse):
    embedding_vector=EMD.embedding_layer(4).embedding(inputSparse)
    print(embedding_vector)
#test4EMD(inputSparse)

#CIN model build test
fieldInput=np.array([[1,2,3,4],[5,6,7,8]])
def test4CINBuild(fieldInput):
    model=CIN.CIN_model()
    single_layer=CIN.CIN_Layer(3)
    model.append(single_layer)
    print(model.layerList)
test4CINBuild(fieldInput)

def test4CINLayer(fieldInput):
    model=CIN.CIN_model()
    model.append(CIN.CIN_Layer(3))
    model.append(CIN.CIN_Layer(4))
    model.append(CIN.CIN_Layer(2))
    model.predict(fieldInput)
#test4CINLayer(fieldInput)

#test4seq
rawInput=np.array([[0,0,0,1,0],[0,0,1],[1,1,0,1,0,1,0]])
def test4seq(rawInput):
    LR_model=LR.LR_model()
    DNN_model_layer=DNN.DNN_model()
    DNN_model_layer.append(DNN.DNN_Layer(3,"relu"))
    DNN_model_layer.append(DNN.DNN_Layer(4,"relu"))
    DNN_model_layer.append(DNN.DNN_Layer(5,"relu"))
    CIN_model=CIN.CIN_model()
    CIN_model.append(CIN.CIN_Layer(3))
    CIN_model.append(CIN.CIN_Layer(4))
    CIN_model.append(CIN.CIN_Layer(2))
    EMD_model=EMD.embedding_layer(4)
    sequence=Sequence.ModelSequence(LR_model,CIN_model,DNN_model_layer,EMD_model)
    result=sequence.predict(rawInput)
    print("model sequence result is {}".format(result))

#test4seq(rawInput)

input=np.array([[1,2,3,4]])
dataTensors_list=[
    input
]
class seqTest(object):
    theta=2
def test4bp():
    model=DNN.DNN_model()
    single_layer=DNN.DNN_Layer(3,"relu")
    model.append(single_layer)
    out_layer=DNN.DNN_Layer(1,"sigmoid")
    model.append(out_layer)
    res,dataTensors_list=model.predict(input)
    # print("******************")
    # print(model.Weights)
    # print(model.biase)
    # print("****************")
    model.paramatersUpdate(0.1,seqTest(),dataTensors_list)
    # print("******************")
    # print(model.Weights)
    # print(model.biase)
    # print("****************")

test4bp()

