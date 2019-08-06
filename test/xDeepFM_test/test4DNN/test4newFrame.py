from xDeepFM import *
#import numpy as np

#LR_model test
input=np.array([[1,2,3,4,5]])
def test4LR(input):
    LR_result=LR.LR_model().predict(input)
    print(LR_result)
    return LR_result
test4LR(input)

#DNN_model test
model=DNN.DNN_model()
single_layer=DNN.DNN_Layer(3,"relu")
model.append(single_layer)
print(model.layerList)
"""
DNN_LAYER test
"""
model_layer=DNN.DNN_model()
model_layer.append(DNN.DNN_Layer(3,"relu"))
model_layer.append(DNN.DNN_Layer(4,"relu"))
model_layer.append(DNN.DNN_Layer(5,"relu"))
model_layer.append(DNN.DNN_Layer(1,"sigmoid"))
#model_layer.build_model(input)
model_layer.predict(input)
print(DNN.activation_function("sigmoid").activation(30))
