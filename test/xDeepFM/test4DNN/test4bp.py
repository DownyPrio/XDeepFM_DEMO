from xDeepFM import *
trainData=np.array([[1,2,3,0],
                    [4,5,6,1],
                    [0,1,2,0]])
model=DNN.DNN_model(2,2)
print(model.weights)
print(model.biase)
ParaUpdata.ParametersUpdate().backPropagation(trainData,model,0.1,10)
print(model.weights)
print(model.biase)