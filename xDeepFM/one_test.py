
import numpy as np

testset=np.array([[1,2]])
parameters=np.array([
    [[1,2],[3,4]],[1,2]

])
print(DNN.DNN_model(1,2).predict(testset))
print(LR.LR_model().predict(testset))